"""Baseline LightGBM model for PS S6E4 - Predicting Irrigation Need."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

CAT_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
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


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    return train, test


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in CAT_COLS:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("category")
    return pdf


def main() -> None:
    train_pl, test_pl = load_data()
    print(f"Train shape: {train_pl.shape}")
    print(f"Test shape:  {test_pl.shape}")

    train = to_pandas(train_pl)
    test = to_pandas(test_pl)

    test_ids = test["id"].to_numpy()
    feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
    X = train[feature_cols]
    y_raw = train[TARGET]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_test = test[feature_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**LGBM_PARAMS)  # type: ignore[arg-type]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )

        oof_proba[val_idx] = model.predict_proba(X_val)
        test_proba += model.predict_proba(X_test) / N_FOLDS

        fold_acc = (oof_proba[val_idx].argmax(1) == y_val).mean()
        print(f"  Fold {fold} accuracy: {fold_acc:.4f}")

    oof_acc = (oof_proba.argmax(1) == y).mean()
    print(f"\nOOF accuracy: {oof_acc:.4f}")

    predictions = le.inverse_transform(test_proba.argmax(1))

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: predictions})
    out_path = SUBMISSIONS_DIR / "baseline_lgbm.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
