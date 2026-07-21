"""Chained binary cascade x te13 x u (Day-17 honest closure).

The Day-9 cascade veto (core-member signature, no LR seat) was measured
pre-TE-era on the repaired surface. Since then the te13 recipe lifted every
consumer by ~+0.001 and the u surface became the deployed lineage — the FTT
precedent (Day-10: pre-TE veto re-measured with te13, still no seat) says
pre-recipe vetoes must be re-earned, not assumption-transferred. This rerun
gives both binary stages the exact current recipe so the target-factorization
axis is closed honestly.

Stage 1: LGBM binary P(at-risk | x), regularized-shallow te3 config.
Stage 2: LGBM binary P(fit | minority, x) on minority rows only.
Features: raw 13 cols (cats as category) + the cached 39-col te13 block
(zoo_common.te13_block_for_fold, tag "u" — same cache the NN consumers hit).
Recombine: kaggle.decision.cascade_combine -> 3-class proba.

Paired read: vs lgbm_te3_u_s42 (0.9503, same features/config, multiclass
objective). Verdict = solo OOF + 12-cell LR seat, escalation gate +0.0004.

Runs on CPU deliberately (LGBM CUDA is a wash on this dataset; the GPU is the
NN lane's): device_type="cpu", n_jobs=-1.

Run: S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_cascade_te3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from train_common import SEEDS, finalize, load_dataset
from train_lgbm_te3 import LGBM_TE3_PARAMS
from zoo_common import clear_ckpt, te13_block_for_fold, zoo_cv

from data_science_stuff.kaggle.decision import cascade_combine

warnings.filterwarnings("ignore", category=FutureWarning)

SURFACE_TAG = "u"


def binary_te3_params() -> dict[str, object]:
    """The translated Susol config adapted to a binary stage, forced to CPU."""
    p = dict(LGBM_TE3_PARAMS)
    p.pop("num_class", None)
    p["objective"] = "binary"
    p["metric"] = "binary_logloss"
    p["device_type"] = "cpu"
    p["n_jobs"] = -1
    return p


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None, "S6E7_REPAIR_FIT not wired for this trainer"
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    params = binary_te3_params()
    print(f"cascade_te3: raw 13 cols + cached te13 block, surface {SURFACE_TAG}")

    def fit_binary(tr_X, tr_y, val_X, val_y, seed):  # noqa: ANN001, ANN202
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            tr_X,
            tr_y,
            eval_set=[(val_X, val_y)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == [0, 1], model.classes_
        return model

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        t13_tr, t13_te, te_names = te13_block_for_fold(
            ds, tr_idx, seed, fold, SURFACE_TAG
        )

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], t13_tr[tr_idx])
        X_va = assemble(X.iloc[val_idx], t13_tr[val_idx])
        X_te = assemble(X_test, t13_te)
        y_tr, y_va = ds.y[tr_idx], ds.y[val_idx]

        # stage 1: at-risk (class 0) vs minority — the live boundary as objective
        m1 = fit_binary(
            X_tr, (y_tr == 0).astype(int), X_va, (y_va == 0).astype(int), seed
        )
        p1_val = m1.predict_proba(X_va)[:, 1]
        p1_test = m1.predict_proba(X_te)[:, 1]

        # stage 2: fit (1) vs unhealthy (2), minority rows only
        min_tr, min_va = y_tr != 0, y_va != 0
        m2 = fit_binary(
            X_tr[min_tr],
            (y_tr[min_tr] == 1).astype(int),
            X_va[min_va],
            (y_va[min_va] == 1).astype(int),
            seed,
        )
        p2_val = m2.predict_proba(X_va)[:, 1]
        p2_test = m2.predict_proba(X_te)[:, 1]

        val_proba = cascade_combine([p1_val, p2_val], [0, 1, 2])
        test_proba = cascade_combine([p1_test, p2_test], [0, 1, 2])
        return val_proba, test_proba

    ckpt = f"cascade_te3_{SURFACE_TAG}_s{SEEDS[0]}"
    oof, test, fold_scores = zoo_cv(ds, fit_fold, ckpt_name=ckpt, seed=SEEDS[0])
    finalize("cascade_te3", ds, oof, test, fold_scores)
    clear_ckpt(ckpt)


if __name__ == "__main__":
    main()
