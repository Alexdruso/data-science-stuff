"""Approach 1: Formula probabilities + 6 raw Deotte features as LGBM input."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from formula import CLASSES, COEFS, INTERCEPTS, build_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

TARGET = "Irrigation_Need"
N_FOLDS = 5
LGBM_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

DEOTTE_FEATURES = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Crop_Growth_Stage",
    "Mulching_Used",
]
CAT_COLS = ["Crop_Growth_Stage", "Mulching_Used"]


def formula_proba(df: pl.DataFrame) -> np.ndarray:
    X = build_features(df)
    logits = INTERCEPTS + X @ COEFS
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def build_lgbm_features(df_pl: pl.DataFrame) -> pd.DataFrame:
    proba = formula_proba(df_pl)
    pdf = df_pl.select(DEOTTE_FEATURES).to_pandas()
    for col in CAT_COLS:
        pdf[col] = pdf[col].astype("category")
    for i, cls in enumerate(CLASSES):
        pdf[f"formula_prob_{cls}"] = proba[:, i]
    return pdf


def main() -> None:
    train_pl = pl.read_csv(DATA_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    X = build_lgbm_features(train_pl)
    X_test = build_lgbm_features(test_pl)
    test_ids = test_pl["id"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(train_pl[TARGET].to_numpy())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**LGBM_PARAMS)  # type: ignore[arg-type]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])

        oof_proba[val_idx] = model.predict_proba(X_val)
        test_proba += model.predict_proba(X_test) / N_FOLDS

        fold_acc = (oof_proba[val_idx].argmax(1) == y_val).mean()
        print(f"  Fold {fold} accuracy: {fold_acc:.4f}")

    oof_acc = (oof_proba.argmax(1) == y).mean()
    print(f"\nOOF accuracy: {oof_acc:.4f}")

    predictions = le.inverse_transform(test_proba.argmax(1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "formula_feature_lgbm.csv"
    pd.DataFrame({"id": test_ids, TARGET: predictions}).to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
