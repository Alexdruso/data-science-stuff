"""HGBC + exact-value numeric TE on the UNREPAIRED surface — Susol v0.7 replica.

Discussion/719199: HistGradientBoosting + cross-fitted exact-value target
encoding of the numerics, plain surface, = OOF 0.9502 -> LB 0.95036 with NO
haircut — +0.0009 over CatBoost-class bases. We only ever measured this TE on
the REPAIRED surface (+0.0002, sub-gate, Day-7). Hypothesis: the R2a remask
deletes ~2x the observed values in the 4 shifted columns, prior-filling the TE
features exactly where their resolution lives — the repair eats the TE gain.

Calibration anchor: OOF ~0.9502 confirms; ~0.9494 (= hgbc_bagw) refutes.

Run (NO S6E7_REPAIR — unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc import HGBC_PARAMS
from train_te_num import encode_fold, with_te
from zoo_common import clear_ckpt, zoo_cv


def main() -> None:
    ds = load_dataset()
    num_cols = [c for c in NUM_COLS if c in ds.train.columns]
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    print(f"HGBC + exact-value TE on {len(num_cols)} numerics, UNREPAIRED surface")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        te_tr, full_maps, prior = encode_fold(X, ds.y, tr_idx, num_cols, seed)
        X_tr = pd.concat([X.iloc[tr_idx], te_tr], axis=1)
        X_va = with_te(X.iloc[val_idx], full_maps, prior)
        X_te = with_te(X_test, full_maps, prior)
        model = HistGradientBoostingClassifier(**HGBC_PARAMS, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te_u_s{SEEDS[0]}")
    _ = np.asarray  # keep numpy import for type checkers


if __name__ == "__main__":
    main()
