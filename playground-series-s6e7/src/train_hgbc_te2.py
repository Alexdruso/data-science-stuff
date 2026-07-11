"""HGBC + sklearn-native TargetEncoder on numerics — exact Susol v0.7 replica.

Our own exact-value TE (fixed pseudo-count 20, train_te_num) gives HGBC only
+0.0002 on the unrepaired surface (hgbc_te_u_s42 0.9496) vs his +0.0009
(OOF 0.9502, LB 0.95036 no haircut). The remaining implementation deltas:
sklearn's TargetEncoder uses per-value EMPIRICAL-BAYES shrinkage
(smooth="auto", variance-based) instead of a fixed pseudo-count — a large
difference for step_count's 12.8k unique values (~50 rows each) — and NaN is
encoded as its own category (our version leaves TE NaN for the trees).

Protocol per outer fold: TargetEncoder(target_type="multiclass", smooth="auto",
cv=5 internal) fit_transform on the fold-train (cross-fitted), transform for
val/test (full-fold encodings). Raw numerics kept alongside; cats stay native.

Calibration anchor: OOF ~0.9502. Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc import HGBC_PARAMS
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    ds = load_dataset()
    num_cols = [c for c in NUM_COLS if c in ds.train.columns]
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    print(f"HGBC + sklearn TargetEncoder (EB auto) on {len(num_cols)} numerics, "
          "UNREPAIRED surface")

    te_names = [f"te_{c}_{k}" for c in num_cols for k in range(N_CLASSES)]

    def with_blocks(base: pd.DataFrame, enc: np.ndarray) -> pd.DataFrame:
        blk = pd.DataFrame(enc, columns=te_names, index=base.index)
        return pd.concat([base, blk], axis=1)

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        te = TargetEncoder(target_type="multiclass", smooth="auto", cv=5)
        enc_tr = te.fit_transform(X[num_cols].iloc[tr_idx], ds.y[tr_idx])
        enc_va = te.transform(X[num_cols].iloc[val_idx])
        enc_te = te.transform(X_test[num_cols])
        X_tr = with_blocks(X.iloc[tr_idx], enc_tr)
        X_va = with_blocks(X.iloc[val_idx], enc_va)
        X_te = with_blocks(X_test, enc_te)
        model = HistGradientBoostingClassifier(**HGBC_PARAMS, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te2_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te2", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te2_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
