"""Optuna hyperparameter search for XGBoost — PS S6E6.

Runs 3-fold CV per trial for speed; saves best params to
results/best_params_xgboost.json. GPU-accelerated (device="cuda").

Imbalance is handled exactly as in train_xgboost.py — inverse-frequency
``sample_weight`` passed to ``.fit()`` (XGBoost has no ``class_weight``). Tuning
without it would optimise for a different regime than the retrain uses.

Run with:  python src/tune_xgboost.py [--n-trials N] [--timeout SECS]
After tuning, re-run src/train_xgboost.py — it auto-loads the JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from train_xgboost import make_sample_weights

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna not installed — run: uv pip install optuna") from e

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_TUNE_FOLDS = 3
# Fixed params not part of the search space (GPU/objective/early stopping stay
# authoritative; train_xgboost.py's XGB_PARAMS.update(tuned) must not collide).
FIXED_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}


def objective(
    trial: "optuna.Trial",
    X: "np.ndarray",
    y: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    params: dict[str, object] = {
        **FIXED_PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=42)
    fold_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = np.argmax(model.predict_proba(X_val), axis=1)
        fold_scores.append(float(balanced_accuracy_score(y_val, val_pred)))

    return float(np.mean(fold_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train_pl = build_features(train_raw)
    train_pl = compute_group_features(train_raw, train_pl)

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    train_pd = train_pl.to_pandas()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_pd[cat_cols] = oe.fit_transform(train_pd[cat_cols])

    X = train_pd[feature_cols].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    sample_weights = make_sample_weights(y)
    print(f"Feature matrix: {X.shape}  classes: {list(le.classes_)}")
    print("XGBoost device: cuda (GPU)")

    study = optuna.create_study(direction="maximize", study_name="xgb_s6e6")
    study.optimize(
        lambda trial: objective(trial, X, y, sample_weights),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: {best.value:.4f}")
    print("Params:", json.dumps(best.params, indent=2))

    out_path = RESULTS_DIR / "best_params_xgboost.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved → {out_path}")
    print("Re-run src/train_xgboost.py to train with tuned params.")


if __name__ == "__main__":
    main()
