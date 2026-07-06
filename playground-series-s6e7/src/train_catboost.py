"""CatBoost model for PS S6E7 — Predicting Student Health Risk.

5-fold MultiClass. CatBoost needs categorical features as strings with no NaN, so
categoricals are filled with a sentinel; numerics keep NaN (handled natively).
Imbalance via auto_class_weights="Balanced" (toggle); decision-weight tuning on
OOF happens in train_common.finalize.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    Dataset,
    bagged_cv,
    finalize,
    load_dataset,
)

CAT_FILL = "__missing__"
# "Balanced" → auto inverse-frequency class weights; None → natural distribution.
CLASS_WEIGHT: object = "Balanced"

# This box has a confirmed working CatBoost GPU build (RTX 2060). A live probe fit
# is unreliable — it transiently fails (and would silently drop the whole run to a
# ~23-min/fold CPU path) if the GPU still holds an allocation from a just-killed
# process. So default to GPU; set CB_TASK_TYPE=CPU to override.
_TASK_TYPE = os.environ.get("CB_TASK_TYPE", "GPU")
# Fixed iteration count — no eval_set/early stopping (see GPU memory note below).
ITERATIONS = 1200
CB_PARAMS: dict[str, object] = {
    "loss_function": "MultiClass",
    "iterations": ITERATIONS,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_state": 42,
    "task_type": _TASK_TYPE,
    "auto_class_weights": CLASS_WEIGHT,
    "verbose": 0,
    # GPU memory rule (RTX 2060, 6 GB). Two independent triggers made CatBoost
    # silently fall back to CPU (~23 min/fold vs ~1 min on GPU):
    #   1. default max_ctr_complexity=2 builds categorical feature-COMBINATION CTR
    #      tables that exceed 6 GB on 690k rows → cap at 1 (no combinations);
    #   2. passing an eval_set holds a second CTR set on the GPU → over 6 GB again,
    #      so we DROP early stopping and train a fixed ITERATIONS instead. The val
    #      split is still predicted afterwards for OOF; it just isn't fed to fit().
    # gpu_ram_part trims the pre-allocated pool. Both GPU-only; ignored on CPU.
    **({"max_ctr_complexity": 1, "gpu_ram_part": 0.85} if _TASK_TYPE == "GPU" else {}),
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params_catboost.json"
    base: dict[str, object] = dict(CB_PARAMS)
    if params_path.exists():
        with params_path.open() as f:
            base.update(json.load(f))
        print(f"Loaded tuned params from {params_path}")
    return base


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = ds.train.copy(), ds.test.copy()
    # CatBoost categoricals must be non-null strings.
    for col in ds.cat_cols:
        X[col] = X[col].fillna(CAT_FILL).astype(str)
        X_test[col] = X_test[col].fillna(CAT_FILL).astype(str)

    params = load_params()
    print(f"CatBoost task_type: {params['task_type']}")
    cat_idx = [X.columns.get_loc(c) for c in ds.cat_cols]
    test_pool = Pool(X_test, cat_features=cat_idx)

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tr_pool = Pool(X.iloc[tr_idx], ds.y[tr_idx], cat_features=cat_idx)
        val_pool = Pool(X.iloc[val_idx], cat_features=cat_idx)  # OOF prediction only
        model = CatBoostClassifier(**{**params, "random_state": seed})
        model.fit(tr_pool)  # no eval_set — keeps CTR tables within the 6 GB GPU
        # CatBoost orders proba columns by sorted class label; y is 0..K-1 so it lines up.
        assert [int(c) for c in model.classes_] == list(range(N_CLASSES)), (
            model.classes_
        )
        return model.predict_proba(val_pool), model.predict_proba(test_pool)

    return bagged_cv(ds, fit_fold)


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}")
    oof_proba, test_proba, fold_scores = run(ds)
    finalize("catboost", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
