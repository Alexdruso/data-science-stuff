"""Baseline LightGBM model for PS S6E7 — Predicting Student Health Risk.

3-class balanced-accuracy task. Trains 5-fold, stores OOF/test probabilities in
CLASSES order, and reports both plain-argmax and decision-weight-tuned OOF
balanced accuracy (see train_common.finalize).
"""

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from lgbm_device import get_lgbm_device
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    Dataset,
    bagged_cv,
    finalize,
    load_dataset,
)

# Imbalance handling. "balanced" was the single biggest lever on the sibling
# balanced-accuracy comp S6E6 (+0.008); decision-weight tuning on OOF adds a bit
# more on top (finalize() always does that). Flip to None for the natural-distribution
# calibration experiment (see CLAUDE.md "Next Steps").
CLASS_WEIGHT: object = "balanced"

_DEVICE_TYPE, _N_JOBS = get_lgbm_device()
LGBM_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": CLASS_WEIGHT,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": _N_JOBS,
    "device_type": _DEVICE_TYPE,
}


def load_params() -> dict[str, object]:
    """Merge Optuna-tuned params (results/best_params_lgbm.json) over the defaults."""
    params_path = RESULTS_DIR / "best_params_lgbm.json"
    base: dict[str, object] = dict(LGBM_PARAMS)
    if params_path.exists():
        with params_path.open() as f:
            base.update(json.load(f))
        print(f"Loaded tuned params from {params_path}")
    base["num_class"] = N_CLASSES
    return base


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    params = load_params()

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        # LGBM predict_proba columns follow model.classes_ (sorted encoded ints 0..K-1
        # == CLASSES order). Assert so a future relabelling can't silently misalign.
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    return bagged_cv(ds, fit_fold)


def main() -> None:
    ds = load_dataset()
    print(
        f"Train: {ds.train.shape}   Test: {ds.test.shape}   classes: {ds.y.max() + 1}"
    )
    oof_proba, test_proba, fold_scores = run(ds)
    finalize("lgbm", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
