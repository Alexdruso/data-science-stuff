"""sklearn HistGradientBoosting model for PS S6E7 — 4th GBDT-family base.

Sourced from a public S6E7 notebook and verified in our harness: fold-1 argmax 0.9494 in
~5s/35 trees — CatBoost-tier out of the box. The differentiators vs our other GBDTs:
early stopping scored on **balanced_accuracy directly** (the competition metric) instead
of log-loss, and sklearn's own histogram/NaN implementation. Trains in seconds per fold,
so the full bagged run is ~2 minutes on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, Dataset, bagged_cv, finalize, load_dataset

HGBC_PARAMS: dict[str, object] = {
    "loss": "log_loss",
    "scoring": "balanced_accuracy",  # early stopping on the competition metric
    "class_weight": "balanced",
    "max_iter": 10000,
    "n_iter_no_change": 10,
    "learning_rate": 0.05,
    "max_depth": None,
    "validation_fraction": 0.1,
    "categorical_features": "from_dtype",
}


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        model = HistGradientBoostingClassifier(**HGBC_PARAMS, random_state=seed)
        model.fit(X.iloc[tr_idx], ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    return bagged_cv(ds, fit_fold)


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}")
    oof_proba, test_proba, fold_scores = run(ds)
    finalize("hgbc", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
