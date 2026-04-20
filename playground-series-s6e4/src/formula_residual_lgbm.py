"""Approach 2: LGBM residual boosting — init_score from formula logits."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
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
    "num_leaves": 127,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
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


def raw_logits(df: pl.DataFrame) -> np.ndarray:
    """Return un-normalized logit scores shape (n, 3) — before softmax."""
    X = build_features(df)
    return INTERCEPTS + X @ COEFS


def to_pandas_deotte(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(DEOTTE_FEATURES).to_pandas()
    for col in CAT_COLS:
        pdf[col] = pdf[col].astype("category")
    return pdf


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    train_pl = pl.read_csv(DATA_DIR / "train.csv")
    test_pl = pl.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    X = to_pandas_deotte(train_pl)
    X_test = to_pandas_deotte(test_pl)
    test_ids = test_pl["id"].to_numpy()

    train_logits = raw_logits(train_pl)
    test_logits = raw_logits(test_pl)

    le = LabelEncoder()
    y = le.fit_transform(train_pl[TARGET].to_numpy())
    # LightGBM multiclass init_score: flattened row-major (n_samples * n_classes,)
    # each row is [score_class0, score_class1, score_class2]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))

    cat_feature_names = [c for c in DEOTTE_FEATURES if c in CAT_COLS]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        init_tr = train_logits[train_idx].ravel(order="C")
        init_val = train_logits[val_idx].ravel(order="C")

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            init_score=init_tr,
            categorical_feature=cat_feature_names,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            init_score=init_val,
            reference=dtrain,
            categorical_feature=cat_feature_names,
            free_raw_data=False,
        )

        params = dict(LGBM_PARAMS)
        n_estimators = int(params.pop("n_estimators"))  # type: ignore[arg-type]
        params.pop("random_state", None)
        params["seed"] = 42

        booster = lgb.train(
            params,  # type: ignore[arg-type]
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.log_evaluation(period=200)],
        )

        # Predictions: booster returns raw margin scores; add init_score then softmax
        raw_val = booster.predict(X_val, raw_score=True) + train_logits[val_idx]
        raw_test = booster.predict(X_test, raw_score=True) + test_logits

        oof_proba[val_idx] = softmax(raw_val)
        test_proba += softmax(raw_test) / N_FOLDS

        fold_acc = (oof_proba[val_idx].argmax(1) == y_val).mean()
        print(f"  Fold {fold} accuracy: {fold_acc:.4f}")

    oof_acc = (oof_proba.argmax(1) == y).mean()
    print(f"\nOOF accuracy: {oof_acc:.4f}")

    predictions = le.inverse_transform(test_proba.argmax(1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "formula_residual_lgbm.csv"
    pd.DataFrame({"id": test_ids, TARGET: predictions}).to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
