"""Optuna hyperparameter search for PS S6E5 — XGBoost pit stop predictor."""

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 3
N_TRIALS = 50
FIXED_PARAMS: dict[str, object] = {
    "n_estimators": 1000,
    "device": "cuda",
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "verbosity": 0,
    "random_state": 42,
}


def prepare_data() -> tuple[pd.DataFrame, np.ndarray]:
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
    return X, y


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
) -> float:
    params = {
        **FIXED_PARAMS,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return float(roc_auc_score(y, oof))


def main() -> None:
    print("Preparing data...")
    X, y = prepare_data()

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
    out = RESULTS_DIR / "best_params_xgboost.json"
    with out.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
