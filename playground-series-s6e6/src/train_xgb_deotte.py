"""Faithful port of cdeotte's XGB v1 (S6E6, notebook OOF 0.96694) to our stack.

Uses deotte_features.build_feature_matrix for the ~240-feature recipe, adds the per-class
FOLD-SAFE TargetEncoder on quantile bins inside the CV loop (sklearn TargetEncoder, binary
per class, smooth=20, cv=5 — the leakage-safe nested encoding that was the big missing piece
in our earlier lgbm_fe attempts), then trains XGBoost-CUDA with cdeotte's exact params and
custom 1−balanced_acc early stopping. Train/test are sorted by id so oof/test arrays match our
id-sorted convention and drop straight into build_lr_stack / build_gbdt_stack.

GATE: standalone OOF argmax should land near 0.9669 (notebook's honest CV). Then add
oof_xgb_deotte/test_xgb_deotte to the stack — first base meaningfully above our 0.9654.

Run:  python src/train_xgb_deotte.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TE_INNER_SPLITS, TE_SMOOTH,
    TOP_FEATURES, build_feature_matrix, cat_key,
)
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
RUN = "xgb_deotte"


def balanced_error_metric(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    # sample_weight is passed by XGBoost (sample_weight_eval_set) but intentionally
    # ignored — early-stop on UNWEIGHTED balanced accuracy, as in cdeotte's notebook.
    yp = y_pred if y_pred.ndim == 2 else y_pred.reshape(-1, len(CLASSES))
    return 1.0 - balanced_accuracy_score(np.asarray(y_true).astype(int), np.argmax(yp, axis=1))


TUNED_PARAMS_PATH = RESULTS_DIR / "best_params_xgb_deotte.json"

CDEOTTE_DEFAULTS = {
    "max_leaves": 72, "min_child_weight": 10, "gamma": 0.20,
    "reg_alpha": 0.30, "reg_lambda": 4.0,
    "subsample": 0.82, "colsample_bytree": 0.74, "colsample_bylevel": 0.86,
}


def make_xgb_params(seed: int, overrides: dict | None = None) -> dict:
    base = {
        "objective": "multi:softprob", "num_class": len(CLASSES),
        "eval_metric": balanced_error_metric, "tree_method": "hist", "device": "cuda",
        "learning_rate": 0.012, "n_estimators": 7000, "early_stopping_rounds": 180,
        "max_depth": 0, "grow_policy": "lossguide", "max_bin": 512,
        **CDEOTTE_DEFAULTS,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    if overrides:
        base.update(overrides)
    return base


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    wpc = np.float32(len(y)) / (np.float32(len(CLASSES)) * np.maximum(counts, 1.0))
    return wpc[y].astype(np.float32)


def sorted_factorize(tr: pd.Series, va: pd.Series, te: pd.Series):
    vals = pd.concat([cat_key(tr), cat_key(va), cat_key(te)], ignore_index=True)
    cats = pd.Index(sorted(vals.unique()))
    codes = vals.map({c: i for i, c in enumerate(cats)}).fillna(-1).astype("int32").to_numpy()
    n1, n2 = len(tr), len(va)
    return codes[:n1], codes[n1:n1 + n2], codes[n1 + n2:]


def te_sources(top: list[str], cat_cols: list[str]) -> list[str]:
    return [c for c in cat_cols if any(f.startswith(f"TE_{c}_") for f in top)]


def add_fold_safe_te(X_tr, y_tr, X_va, X_te, te_cols):
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        ktr, kva, kte = cat_key(X_tr[c]).to_frame(c), cat_key(X_va[c]).to_frame(c), cat_key(X_te[c]).to_frame(c)
        for cls_idx, cls_name in INT_TO_CLASS.items():
            yb = (y_tr == cls_idx).astype(np.int32)
            enc = TargetEncoder(target_type="binary", smooth=TE_SMOOTH, cv=TE_INNER_SPLITS,
                                shuffle=True, random_state=SEED + 177)
            name = f"TE_{c}_{cls_name}"
            X_tr[name] = enc.fit_transform(ktr, yb).ravel().astype(np.float32)
            X_va[name] = enc.transform(kva).ravel().astype(np.float32)
            X_te[name] = enc.transform(kte).ravel().astype(np.float32)
    return X_tr, X_va, X_te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-tuned", action="store_true",
                    help=f"load optimised params from {TUNED_PARAMS_PATH}")
    args = ap.parse_args()

    param_overrides: dict | None = None
    if args.use_tuned:
        if not TUNED_PARAMS_PATH.exists():
            raise FileNotFoundError(f"No tuned params found at {TUNED_PARAMS_PATH}. "
                                    "Run tune_xgb_deotte.py first.")
        with open(TUNED_PARAMS_PATH) as f:
            tuned = json.load(f)
        param_overrides = tuned["params"]
        print(f"★ Using tuned params (best 3-fold {tuned['best_value']:.5f} "
              f"vs baseline {tuned['baseline']:.5f}): {param_overrides}")
    global RUN
    if args.use_tuned:
        RUN = "xgb_deotte_tuned"

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)  # drop -9999 corruption row
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    print(f"X {X.shape}  cat_cols {len(cat_cols)}  TE sources {len(TE_COLS)}  model cats {len(MODEL_CAT_COLS)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        if fold == 1:
            print(f"  features used: {len(feats)}/{len(TOP_FEATURES)} "
                  f"(missing {len(TOP_FEATURES) - len(feats)})")
        Xtr = X_tr[feats].astype(np.float32).to_numpy()
        Xva = X_va[feats].astype(np.float32).to_numpy()
        Xte = X_te[feats].astype(np.float32).to_numpy()

        sw = class_weights(y[tri])
        vw = class_weights(y[vai])
        model = XGBClassifier(**make_xgb_params(SEED + fold * 100, param_overrides))
        model.fit(Xtr, y[tri], sample_weight=sw, eval_set=[(Xva, y[vai])],
                  sample_weight_eval_set=[vw], verbose=False)
        oof[vai] = model.predict_proba(Xva)
        test_proba += model.predict_proba(Xte) / N_SPLITS
        best_iters.append(int(getattr(model, "best_iteration", 0) or 0))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  (best_iter {best_iters[-1]})")
        del model, X_tr, X_va, X_te, Xtr, Xva, Xte

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [notebook 0.96694; our best base LGBM 0.9654]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels, id_col=ID_COL)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
