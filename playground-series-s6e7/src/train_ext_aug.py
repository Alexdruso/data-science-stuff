"""External-data augmentation probe for PS S6E7 — inject the original 50k rows as signal.

`data/external/student_health_dataset_50k.csv` has the competition's EXACT schema (same 13
features, same categorical levels, near-identical class balance) — almost certainly the real
source the synthetic competition data was generated from. It has ZERO missing values, so every
external row is a complete-driver row; whether that transfers is the open question the gate
answers (the clean original may sit closer to the test distribution than the noisy train does).

Augmentation, NOT a join: external rows are concatenated into each TRAINING fold only (never
validated on), with a tunable sample weight (S6E7_EXT_W, default 1.0). OOF is measured purely on
the real competition validation rows, so the CV/gate stay honest. Reuses bagged_cv — fit_fold is
free to build whatever training matrix it likes as long as it returns (comp_val_proba, test_proba).
Then gate oof_lgbm_ext with diag_mlp_transfer.py (must lift adv-weighted OOF, clear margin).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from features import CLASSES, TARGET, build_features
from train_common import (
    DATA_DIR,
    N_CLASSES,
    Dataset,
    bagged_cv,
    finalize,
    load_dataset,
)

EXT_W: float = float(os.environ.get("S6E7_EXT_W", "1.0"))
EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"


def load_external(ds: Dataset) -> tuple[object, np.ndarray]:
    """External features aligned to ds.feature_cols + encoded target (CLASSES order)."""
    # External is training-augmentation only (never OOF-aligned), so no build_features
    # sort — and it keys on student_id, not the competition's id column.
    ext = pl.read_csv(EXT_CSV).to_pandas()
    for c in ds.feature_cols:
        assert c in ext.columns, f"external missing feature {c}"
    y_ext = ext[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X_ext = ext[ds.feature_cols].copy()
    for c in ds.cat_cols:
        X_ext[c] = X_ext[c].astype("category")
    return X_ext, y_ext.astype(np.int64)


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    X_ext, y_ext = load_external(ds)
    # Unify categorical dtypes so concat keeps a single shared category set per column.
    import pandas as pd

    for col in ds.cat_cols:
        cats = X[col].cat.categories.union(X_ext[col].cat.categories)
        X[col] = X[col].cat.set_categories(cats)
        X_ext[col] = X_ext[col].cat.set_categories(cats)
        X_test[col] = X_test[col].astype("category").cat.set_categories(cats)
    print(f"external: {X_ext.shape} rows, weight {EXT_W}")

    params = load_params()

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        X_tr = pd.concat([X.iloc[tr_idx], X_ext], ignore_index=True)
        y_tr = np.concatenate([ds.y[tr_idx], y_ext])
        sw = np.concatenate([np.ones(len(tr_idx)), np.full(len(y_ext), EXT_W)])
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw,
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],  # real rows only for early stop
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    return bagged_cv(ds, fit_fold, seeds=[42])  # single seed for a fast first read


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}")
    oof_proba, test_proba, fold_scores = run(ds)
    finalize("lgbm_ext", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
