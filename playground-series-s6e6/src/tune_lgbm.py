"""Optuna hyperparameter search for LightGBM — PS S6E6.

Runs 3-fold CV per trial for speed; saves best params to results/best_params.json.
Run with:  python src/tune_lgbm.py [--n-trials N] [--timeout SECS]

After tuning, re-run src/baseline.py — it auto-loads results/best_params.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from lgbm_device import get_lgbm_device

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna not installed — run: uv pip install optuna") from e

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_TUNE_FOLDS = 3
_DEVICE_TYPE, _N_JOBS = get_lgbm_device()
# Fixed params that are not part of the search space
FIXED_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": _N_JOBS,
    "device_type": _DEVICE_TYPE,
}


def objective(
    trial: "optuna.Trial",
    X: pd.DataFrame,
    y: np.ndarray,
) -> float:
    params: dict[str, object] = {
        **FIXED_PARAMS,
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }

    skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=42)
    fold_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        val_proba = model.predict_proba(X_val)
        val_pred = np.argmax(val_proba, axis=1)
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
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")

    X = train_pd[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"Feature matrix: {X.shape}  classes: {list(le.classes_)}")
    print(f"LGBM device: {_DEVICE_TYPE}  n_jobs: {_N_JOBS}")

    study = optuna.create_study(direction="maximize", study_name="lgbm_s6e6")
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: {best.value:.4f}")
    print("Params:", json.dumps(best.params, indent=2))

    out_path = RESULTS_DIR / "best_params.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved → {out_path}")
    print("Re-run src/baseline.py to train with tuned params.")


if __name__ == "__main__":
    main()
