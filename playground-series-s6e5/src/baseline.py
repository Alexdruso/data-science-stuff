"""Baseline LightGBM model for PS S6E5 - Predicting F1 Pit Stops."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5
LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params.json"
    base: dict[str, object] = dict(LGBM_PARAMS)
    if params_path.exists():
        with params_path.open() as f:
            tuned = json.load(f)
        base.update(tuned)
        print(f"Loaded tuned params from {params_path}")
    return base


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    return train, test


def to_pandas(df: pl.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in cat_cols:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("category")
    return pdf


def main() -> None:
    train_pl_raw, test_pl = load_data()
    train_pl = build_features(train_pl_raw)
    test_pl = build_features(test_pl)
    train_pl = compute_group_features(train_pl_raw, train_pl)
    test_pl = compute_group_features(train_pl_raw, test_pl)
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    cat_cols = [
        c
        for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train_pl.columns if c not in ("id", TARGET)]

    train = to_pandas(train_pl, cat_cols)
    test = to_pandas(test_pl, cat_cols)

    X = train[feature_cols]
    y = train[TARGET].to_numpy()
    X_test = test[feature_cols]
    test_ids = test["id"].to_numpy()

    params = load_params()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}")

    save_cv_result(RESULTS_DIR, "baseline_lgbm_v5", fold_aucs, oof_auc)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "baseline_lgbm_v5.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
