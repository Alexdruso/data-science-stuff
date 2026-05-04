"""Optuna hyperparameter search for PS S6E5 — LightGBM pit stop predictor."""

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 3
N_TRIALS = 50
FIXED_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train = build_features(train_raw)
    train = compute_group_features(train_raw, train)

    cat_cols = [
        c
        for c in train.columns
        if train[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train.columns if c not in ("id", TARGET)]

    pdf = train.to_pandas()
    for col in cat_cols:
        pdf[col] = pdf[col].astype("category")

    X = pdf[feature_cols]
    y = pdf[TARGET].to_numpy()
    return X, y, cat_cols


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
) -> float:
    params = {
        **FIXED_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return float(roc_auc_score(y, oof))


def main() -> None:
    print("Preparing data...")
    X, y, _ = prepare_data()

    print(f"Running {N_TRIALS} Optuna trials ({N_FOLDS}-fold CV each)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest {N_FOLDS}-fold AUC: {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "best_params.json"
    with out.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
