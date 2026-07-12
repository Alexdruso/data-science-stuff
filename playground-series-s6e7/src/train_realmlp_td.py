"""RealMLP-TD + the deferred NN input recipe (zoo Z1, PS S6E7).

The repo's explicitly deferred item: hand a smooth net the features trees get
free from splits — rule-combo TE + exact-value numeric TE + ordinal scalars
(zoo_common.te_block_for_fold) — turning "learn a discontinuous 3-feature rule"
into "learn a near-linear map". Backbone: data_science_stuff.kaggle.models
RealMLP_TD_Classifier (balanced-softmax loss_prior_power=1.075 default).

Inputs per fold: cat integer codes (0=missing, all cardinalities <=10 so the
model one-hots them), median-imputed numerics + full missingness block (NNData),
TE block (prior-filled) + ordinals (median-imputed). train_bs=512 + fused AdamW
+ AMP keep a fold ~15 min on the 6 GB card.

Run (zoo protocol, repaired surface):
  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_realmlp_td.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_mlp import NNData
from zoo_common import clear_ckpt, te13_block_for_fold, te_block_for_fold, zoo_cv

from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

# TE cache tag must identify the training surface: TE maps depend on which rows
# are NaN, so the m100 (_r) and m050 (_r2) lineages must never share cache entries.
_MULT = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))
# S6E7_RULE=1 (Day-9): corrected-recipe variant — TRUE rule triple TE (_combo2,
# sleep_DURATION buckets x stress x activity) + rule_set one-hots on top of the
# Z1 recipe. Distinct model name + TE cache tag; everything else identical so the
# read vs realmlp_r_s<seed> is paired.
RULE = os.environ.get("S6E7_RULE", "") not in ("", "0")
# S6E7_TE13=1 (Day-9 final block): append the v0.7/te3 all-13 string EB TE block
# (zoo_common.te13_block_for_fold, 39 cols) to the recipe inputs — the block that
# took HGBC to 0.9504. Additive: everything else identical, read is paired.
TE13 = os.environ.get("S6E7_TE13", "") not in ("", "0")
# S6E7_KNN=1 (Day-10 PM): append the kNN label-distribution block
# (knn_features.knn_block_for_fold, 9 cols — joint-density statistic in the 6
# leakage numerics). Flat for the 33-leaf HGBC; the smooth consumer is the
# remaining test of the channel. Additive: read is paired vs realmlp_te13.
KNN = os.environ.get("S6E7_KNN", "") not in ("", "0")
# "u" = UNREPAIRED surface (Day-9 evening, discussion/719199): REPAIR=0 must not
# reuse the "r" TE cache — those blocks were computed on the remasked matrix.
_REPAIR = os.environ.get("S6E7_REPAIR", "") not in ("", "0")
_BASE_TAG = ("r" if _MULT == 1.0 else "r2") if _REPAIR else "u"
SURFACE_TAG = _BASE_TAG + ("_rule" if RULE else "")
MODEL_NAME = (
    ("realmlp_rule" if RULE else "realmlp")
    + ("_te13" if TE13 else "")
    + ("_knn" if KNN else "")
)
CFG = dict(
    train_bs=512,
    fused_optimizer=True,
    allow_amp=True,
    verbosity=1,
)


def main() -> None:
    ds = load_dataset()
    nd = NNData(ds)
    print(
        f"cats {dict(zip(nd.cat_cols, nd.cardinalities))}  device cuda={torch.cuda.is_available()}"
    )

    cat_names = [f"cat_{c}" for c in nd.cat_cols]
    miss_names = [f"miss_{i}" for i in range(nd.miss_tr.shape[1])]

    def frame(
        cat: np.ndarray, num_filled: np.ndarray, miss: np.ndarray, te: np.ndarray
    ) -> pd.DataFrame:
        blocks = {
            **{n: cat[:, i] for i, n in enumerate(cat_names)},
            **{n: num_filled[:, i] for i, n in enumerate(nd.num_cols)},
            **{n: miss[:, i] for i, n in enumerate(miss_names)},
            **{f"te_{i}": te[:, i] for i in range(te.shape[1])},
        }
        return pd.DataFrame(blocks)

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)  # noqa: NPY002 — library-internal RNG expects it

        te_tr_full, te_te, _names = te_block_for_fold(
            ds, tr_idx, seed, fold, SURFACE_TAG, rule=RULE
        )
        if TE13:
            t13_tr, t13_te, _n13 = te13_block_for_fold(
                ds, tr_idx, seed, fold, _BASE_TAG
            )
            te_tr_full = np.concatenate([te_tr_full, t13_tr], axis=1)
            te_te = np.concatenate([te_te, t13_te], axis=1)
        if KNN:
            from knn_features import knn_block_for_fold

            k_tr, k_te, _nk = knn_block_for_fold(ds, tr_idx, seed, fold, _BASE_TAG)
            te_tr_full = np.concatenate([te_tr_full, k_tr], axis=1)
            te_te = np.concatenate([te_te, k_te], axis=1)
        # median-impute raw numerics + the ordinal tail of the TE block (train-fold stats)
        num_med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
        num_tr = np.where(np.isnan(nd.Xnum_tr), num_med, nd.Xnum_tr)
        num_te = np.where(np.isnan(nd.Xnum_te), num_med, nd.Xnum_te)
        te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
        te_tr_f = np.where(np.isnan(te_tr_full), te_med, te_tr_full)
        te_te_f = np.where(np.isnan(te_te), te_med, te_te)

        X_all = frame(nd.Xcat_tr, num_tr, nd.miss_tr, te_tr_f)
        X_test = frame(nd.Xcat_te, num_te, nd.miss_te, te_te_f)
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]

        model = RealMLP_TD_Classifier(**{**CFG, "random_state": fold_seed})
        model.fit(
            X_tr,
            ds.y[tr_idx],
            X_val,
            ds.y[val_idx],
            cat_col_names=cat_names,
            X_test=X_test,
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        val_proba = model.best_val_probs_.astype(np.float64)
        test_proba = model.predict_proba(X_test).astype(np.float64)
        # normalize (model returns softmax means; guard finalize's sum-to-1 assert)
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        test_proba /= test_proba.sum(axis=1, keepdims=True)

        del model, X_all, X_test, X_tr, X_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"{MODEL_NAME}_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize(MODEL_NAME, ds, oof, test, fold_scores)
    clear_ckpt(f"{MODEL_NAME}_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
