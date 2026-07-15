"""CatBoost + the te13 block (Day-12 PM): the mechanism table's missing 4th row.

Day-9 recorded the config×TE lift as "uniform across all four GBDT libraries"
but only HGBC/LGBM/XGB were ever run — CatBoost stayed on the raw recipe
(catboost_bagw 0.9492, an lrstack_u3 leg). This measures the te13 block through
the 4th library: cached per-fold te13 columns (39 numeric) appended to the raw
features, proven GPU config otherwise UNchanged (no config translation — the
Lossguide/min_data_in_leaf combo is a known GPU-trap surface on 6 GB; the read
is te13-features-through-CatBoost, paired vs the same-config raw-recipe runs).

Run (unrepaired namespace, single seed, ckpt-safe):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_catboost_te3.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent))
from train_catboost import CAT_FILL, CB_PARAMS
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from zoo_common import clear_ckpt, te13_block_for_fold, zoo_cv

_REPAIR = os.environ.get("S6E7_REPAIR", "") not in ("", "0")
_MULT = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))
SURFACE_TAG = ("r" if _MULT == 1.0 else "r2") if _REPAIR else "u"


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None, "S6E7_REPAIR_FIT not wired for this trainer"
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna(CAT_FILL).astype(str)
        X_test[col] = X_test[col].fillna(CAT_FILL).astype(str)
    cat_idx = [X.columns.get_loc(c) for c in ds.cat_cols]

    params = dict(CB_PARAMS)
    print(f"CatBoost te13 ({SURFACE_TAG}), task_type: {params['task_type']}")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        t13_tr, t13_te, te_names = te13_block_for_fold(
            ds, tr_idx, seed, fold, SURFACE_TAG
        )
        te_names = [str(n) for n in te_names]

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [
                    base.reset_index(drop=True),
                    pd.DataFrame(blk, columns=te_names),
                ],
                axis=1,
            )

        tr_pool = Pool(
            assemble(X.iloc[tr_idx], t13_tr[tr_idx]),
            ds.y[tr_idx],
            cat_features=cat_idx,
        )
        val_pool = Pool(assemble(X.iloc[val_idx], t13_tr[val_idx]), cat_features=cat_idx)
        test_pool = Pool(assemble(X_test, t13_te), cat_features=cat_idx)
        model = CatBoostClassifier(**{**params, "random_state": seed})
        model.fit(tr_pool)  # no eval_set — keeps CTR tables within the 6 GB GPU
        assert [int(c) for c in model.classes_] == list(range(N_CLASSES)), (
            model.classes_
        )
        val_proba = model.predict_proba(val_pool)
        test_proba = model.predict_proba(test_pool)
        del model, tr_pool, val_pool, test_pool
        return val_proba, test_proba

    tag = os.environ.get("S6E7_RUN_TAG", f"_{SURFACE_TAG}_s{SEEDS[0]}")
    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"catboost_te3{tag}", seed=SEEDS[0]
    )
    finalize("catboost_te3", ds, oof, test, fold_scores)
    clear_ckpt(f"catboost_te3{tag}")


if __name__ == "__main__":
    main()
