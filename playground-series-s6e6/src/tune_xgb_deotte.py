"""Optuna HPO for xgb_deotte on cdeotte's 240-feature pipeline (S6E6).

The deotte bases (xgb_deotte, catboost_v3, realmlp_deotte) carry the 0.9704 stack but use
cdeotte's HARDCODED params — never re-tuned on our 5-fold CV. This sweep tests whether our
setup (RTX 2060 6GB, pandas pipeline, no RAPIDS) has a different optimum.

Design:
- Full deotte feature matrix + fold-safe TargetEncoder (same as train_xgb_deotte.py)
- 3-fold CV (fast inner loop) with a fast-converge config: lr=0.05, n_est=2500, es=60
  → each trial ~3-6 min GPU; 30 trials ≈ 1.5-3 h total
- Search space: ±30-50% around cdeotte's params (structural + regularisation)
- Objective: mean 3-fold argmax balanced_acc (threshold optimisation costs too much per trial)
- After HPO: best params saved to results/best_params_xgb_deotte.json
  → run train_xgb_deotte.py --use-tuned to retrain full 5-fold with optimised params

Gate: if best 3-fold score > cdeotte's 3-fold equivalent (~0.964 ± 0.001), file the params
and retrain. If flat (< +0.0005) → strength axis confirmed at ceiling for xgb_deotte.

Run:  python src/tune_xgb_deotte.py [--n-trials N] [--timeout SECS]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TE_INNER_SPLITS, TE_SMOOTH,
    TOP_FEATURES, build_feature_matrix, cat_key,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise SystemExit("optuna not installed — run: uv pip install optuna") from exc

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = 42
N_TUNE_FOLDS = 3

# Fast-converge fixed params for the inner search loop.
# lr×n_est product comparable to cdeotte's 0.012×7000 = 84 (0.05×1700≈85).
# Early stopping at 60 means a trial exits if no improvement after 60 trees.
FIXED_TUNE = {
    "objective": "multi:softprob",
    "num_class": len(CLASSES),
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "grow_policy": "lossguide",
    "max_bin": 512,
    "learning_rate": 0.05,
    "n_estimators": 2500,
    "early_stopping_rounds": 60,
    "random_state": SEED,
    "n_jobs": 4,
    "verbosity": 0,
}

# cdeotte's reference params (our current baseline)
CDEOTTE_PARAMS = {
    "max_leaves": 72,
    "min_child_weight": 10,
    "gamma": 0.20,
    "reg_alpha": 0.30,
    "reg_lambda": 4.0,
    "subsample": 0.82,
    "colsample_bytree": 0.74,
    "colsample_bylevel": 0.86,
}


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    wpc = np.float32(len(y)) / (np.float32(len(CLASSES)) * np.maximum(counts, 1.0))
    return wpc[y].astype(np.float32)


def te_sources(cat_cols: list[str]) -> list[str]:
    return [c for c in cat_cols if any(f.startswith(f"TE_{c}_") for f in TOP_FEATURES)]


def add_fold_safe_te(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, X_te: pd.DataFrame,
    te_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        ktr = cat_key(X_tr[c]).to_frame(c)
        kva = cat_key(X_va[c]).to_frame(c)
        kte = cat_key(X_te[c]).to_frame(c)
        for cls_idx, cls_name in INT_TO_CLASS.items():
            yb = (y_tr == cls_idx).astype(np.int32)
            enc = TargetEncoder(
                target_type="binary", smooth=TE_SMOOTH, cv=TE_INNER_SPLITS,
                shuffle=True, random_state=SEED + 177,
            )
            name = f"TE_{c}_{cls_name}"
            X_tr[name] = enc.fit_transform(ktr, yb).ravel().astype(np.float32)
            X_va[name] = enc.transform(kva).ravel().astype(np.float32)
            X_te[name] = enc.transform(kte).ravel().astype(np.float32)
    return X_tr, X_va, X_te


def sorted_factorize(
    tr: pd.Series, va: pd.Series, te: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = pd.concat([cat_key(tr), cat_key(va), cat_key(te)], ignore_index=True)
    cats = pd.Index(sorted(vals.unique()))
    codes = vals.map({c: i for i, c in enumerate(cats)}).fillna(-1).astype("int32").to_numpy()
    n1, n2 = len(tr), len(va)
    return codes[:n1], codes[n1:n1 + n2], codes[n1 + n2:]


def run_cv(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
    cat_cols: list[str], params: dict,
) -> float:
    te_cols = te_sources(cat_cols)
    model_cat_cols = [c for c in cat_cols if c in TOP_FEATURES]
    skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores: list[float] = []
    for fold_idx, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        y_tr = y[tri]
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y_tr, X_va, X_te, te_cols)
        for c in model_cat_cols:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        Xtr = X_tr[feats].astype(np.float32).to_numpy()
        Xva = X_va[feats].astype(np.float32).to_numpy()
        sw = class_weights(y_tr)
        model = XGBClassifier(**params)
        model.fit(Xtr, y_tr, sample_weight=sw, eval_set=[(Xva, y[vai])], verbose=False)
        preds = np.argmax(model.predict_proba(Xva), axis=1)
        fold_score = float(balanced_accuracy_score(y[vai], preds))
        best_iter = int(getattr(model, "best_iteration", 0) or 0)
        scores.append(fold_score)
        print(f"    fold {fold_idx}/{N_TUNE_FOLDS}  ba={fold_score:.5f}  iter={best_iter}")
        del model, X_tr, X_va, X_te, Xtr, Xva
    return float(np.mean(scores))


def make_params(trial: "optuna.Trial") -> dict:
    return {
        **FIXED_TUNE,
        # structural — cdeotte's values are the centres
        "max_leaves": trial.suggest_int("max_leaves", 48, 96),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 20.0, log=True),
        # regularisation
        "gamma": trial.suggest_float("gamma", 0.05, 0.60),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.5, 8.0),
        # sampling — tight range, these are well-established
        "subsample": trial.suggest_float("subsample", 0.70, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.90),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.75, 1.00),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    print(f"Feature matrix: {X.shape}  cat_cols: {len(cat_cols)}  classes: {CLASSES}")

    # Baseline: cdeotte's hardcoded params (sanity-check the 3-fold score)
    print("\n── sanity-check: cdeotte's hardcoded params (3-fold) ──")
    baseline_params = {**FIXED_TUNE, **CDEOTTE_PARAMS, "random_state": SEED}
    baseline_score = run_cv(X, y, X_test, cat_cols, baseline_params)
    print(f"  cdeotte baseline 3-fold mean ba: {baseline_score:.5f}\n")

    def objective(trial: "optuna.Trial") -> float:
        params = make_params(trial)
        params["random_state"] = SEED + trial.number * 7
        print(f"\nTrial {trial.number}: {trial.params}")
        score = run_cv(X, y, X_test, cat_cols, params)
        print(f"  → trial {trial.number} mean ba: {score:.5f}  (baseline {baseline_score:.5f}  Δ {score-baseline_score:+.5f})")
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_deotte_s6e6",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    # Seed with cdeotte's params as trial 0 (guarantees baseline is always in the study)
    study.enqueue_trial(CDEOTTE_PARAMS)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial #{best.number}: {best.value:.5f}  (baseline {baseline_score:.5f}  Δ {best.value - baseline_score:+.5f})")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    out = RESULTS_DIR / "best_params_xgb_deotte.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump({"best_value": best.value, "baseline": baseline_score, "params": best.params}, f, indent=2)
    print(f"Saved → {out}")

    delta = best.value - baseline_score
    if delta > 0.0005:
        print(f"\n★ MEANINGFUL GAIN ({delta:+.5f}) — re-run train_xgb_deotte.py --use-tuned for full 5-fold")
    else:
        print(f"\n→ FLAT (Δ={delta:+.5f} < 0.0005) — strength axis confirmed at ceiling for xgb_deotte params")


if __name__ == "__main__":
    main()
