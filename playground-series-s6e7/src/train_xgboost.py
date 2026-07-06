"""XGBoost model for PS S6E7 — Predicting Student Health Risk.

5-fold, multi:softprob, native categorical + NaN handling. GPU if available.
Imbalance handled with balanced sample weights (toggle); decision-weight tuning
on OOF happens in train_common.finalize.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    Dataset,
    bagged_cv,
    finalize,
    load_dataset,
)

# "balanced" → inverse-frequency per-row sample weights; None → natural distribution.
CLASS_WEIGHT: object = "balanced"


def get_device() -> str:
    """Return 'cuda' if an XGBoost CUDA build + GPU works here, else 'cpu'."""
    try:
        import xgboost as xgb

        dtr = xgb.DMatrix(np.zeros((8, 2)), label=np.array([0, 1, 2, 0, 1, 2, 0, 1]))
        xgb.train(
            {
                "tree_method": "hist",
                "device": "cuda",
                "num_class": 3,
                "objective": "multi:softprob",
            },
            dtr,
            num_boost_round=1,
        )
        return "cuda"
    except Exception:  # noqa: BLE001 — any failure → CPU
        return "cpu"


_DEVICE = get_device()
XGB_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": _DEVICE,
    "enable_categorical": True,
    "random_state": 42,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params_xgboost.json"
    base: dict[str, object] = dict(XGB_PARAMS)
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
    print(f"XGB device: {params['device']}")

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sw = (
            compute_sample_weight("balanced", ds.y[tr_idx])
            if CLASS_WEIGHT == "balanced"
            else None
        )
        model = XGBClassifier(**{**params, "random_state": seed})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            sample_weight=sw,
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            verbose=False,
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    return bagged_cv(ds, fit_fold)


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}")
    oof_proba, test_proba, fold_scores = run(ds)
    finalize("xgboost", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
