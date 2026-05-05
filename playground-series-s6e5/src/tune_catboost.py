"""Optuna hyperparameter search for PS S6E5 — CatBoost pit stop predictor."""

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
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
    "iterations": 1000,
    "eval_metric": "AUC",
    "use_best_model": True,
    "early_stopping_rounds": 50,
    "task_type": "GPU",
    "verbose": 0,
    "random_seed": 42,
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

    # CatBoost takes string columns as-is — no category dtype conversion
    pdf = train.to_pandas()
    X = pdf[feature_cols]
    y = pdf[TARGET].to_numpy()
    return X, y, cat_cols


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
) -> float:
    params = {
        **FIXED_PARAMS,
        "cat_features": cat_cols,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return float(roc_auc_score(y, oof))


def main() -> None:
    print("Preparing data...")
    X, y, cat_cols = prepare_data()

    print(f"Running {N_TRIALS} Optuna trials ({N_FOLDS}-fold CV each)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, cat_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest {N_FOLDS}-fold AUC: {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "best_params_catboost.json"
    with out.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
