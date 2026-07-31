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

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_mlp import NNData
from zoo_common import _kseg, clear_ckpt, te13_block_for_fold, te_block_for_fold, zoo_cv

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
# S6E7_IMP=1 (Day-17 P1 escalation): append the driver-reconstruction block
# (imp_features.imp_block_for_fold, 2 cols — fold-safe HGBR sleep imputation).
# Replicated +0.00015 solo on the tree canary but seat veto #14 there; the
# smooth consumer is the remaining test (completed driver = the rule's own
# coordinate). Additive: read is paired vs realmlp_te13.
IMP = os.environ.get("S6E7_IMP", "") not in ("", "0")
# S6E7_DP=1 (Day-23): append the label-free DRIVER-POSTERIOR block
# (dp_features.dp_block, 20 cols — P(driver|x_obs) for all three drivers + the three
# rule-band masses of the sleep posterior). Distinct from S6E7_IMP, which reconstructs
# only sleep_duration and only as a scalar point estimate; activity is the most
# recoverable driver (Day-22) and had never been fed to a strong leg. Additive:
# everything else identical, so the read is paired vs realmlp_te13.
DP = os.environ.get("S6E7_DP", "") not in ("", "0")
# S6E7_ESEL=bll (Day-10 eve): epoch selection by balanced LOGLOSS instead of
# balanced accuracy — the field verified BA-as-selection is noise-prone; our
# strongest leg has always epoch-selected on BA. S6E7_SNAP=k: average softmax
# over the top-k epochs (window average of the steep trajectory). Paired arms.
ESEL = os.environ.get("S6E7_ESEL", "bacc")
SNAP = int(os.environ.get("S6E7_SNAP", "0"))
# "u" = UNREPAIRED surface (Day-9 evening, discussion/719199): REPAIR=0 must not
# reuse the "r" TE cache — those blocks were computed on the remasked matrix.
# "rt" (Day-13, S6E7_REPAIR_FIT=1) = repair-at-fit-only: model/TE fit on the
# remasked matrix, val/test inputs plain — surface-consistent OOF for the joint
# u+r stack. Distinct cache tag: fold-train TE encodings differ from both u and r.
_REPAIR = os.environ.get("S6E7_REPAIR", "") not in ("", "0")
_REPAIR_FIT = os.environ.get("S6E7_REPAIR_FIT", "") not in ("", "0")
_BASE_TAG = ("r" if _MULT == 1.0 else "r2") if _REPAIR else ("rt" if _REPAIR_FIT else "u")
# the "rt" cache tag does not encode the mult (Day-13 audit): forbid the combo
# rather than silently serve mult-1.0 cache entries to a mult!=1 remask.
assert not (_REPAIR_FIT and _MULT != 1.0), "REPAIR_FIT requires REPAIR_MULT=1.0"
SURFACE_TAG = _BASE_TAG + ("_rule" if RULE else "")
MODEL_NAME = (
    ("realmlp_rule" if RULE else "realmlp")
    + ("_te13" if TE13 else "")
    + ("_knn" if KNN else "")
    + ("_imp" if IMP else "")
    + ("_dp" if DP else "")
    + ("_bll" if ESEL == "bll" else "")
    + (f"_snap{SNAP}" if SNAP else "")
)
CFG = dict(
    train_bs=512,
    fused_optimizer=True,
    allow_amp=True,
    verbosity=1,
    epoch_select=ESEL,
    snapshot_k=SNAP,
)
# S6E7_REALMLP_PARAMS (Day-12 tuning): JSON dict merged into CFG for
# confirmation runs of tune_realmlp_td.py winners. MODEL_NAME does NOT change,
# so an explicit fresh S6E7_RUN_TAG is required — never overwrite the
# finals-lineage arrays.
_PARAMS_ENV = os.environ.get("S6E7_REALMLP_PARAMS")
if _PARAMS_ENV:
    if not os.environ.get("S6E7_RUN_TAG"):
        raise SystemExit("S6E7_REALMLP_PARAMS requires an explicit S6E7_RUN_TAG")
    CFG.update(json.loads(_PARAMS_ENV))
    print(f"S6E7_REALMLP_PARAMS override: {json.loads(_PARAMS_ENV)}")


def main() -> None:
    ds = load_dataset()
    nd = NNData(ds)
    nd_fit = None
    if ds.train_fit is not None:
        # DP: the cached posteriors were fitted on the PLAIN (u) matrix, so under
        # repair-at-fit-only they would serve u-surface driver posteriors against
        # remasked fold-train rows — a silent surface mismatch. Forbid the combo.
        assert not (RULE or KNN or DP), "S6E7_REPAIR_FIT not wired for rule/knn/dp recipes"
        from dataclasses import replace

        nd_fit = NNData(replace(ds, train=ds.train_fit, train_fit=None))
        assert nd_fit.cardinalities == nd.cardinalities, "fit/plain cat vocab drifted"
        print("repair-at-fit-only: fold-train rows remasked, val/test plain")
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
            ds, tr_idx, seed, fold, SURFACE_TAG, rule=RULE, fit_df=ds.train_fit
        )
        if TE13:
            t13_tr, t13_te, _n13 = te13_block_for_fold(
                ds, tr_idx, seed, fold, _BASE_TAG, fit_df=ds.train_fit
            )
            te_tr_full = np.concatenate([te_tr_full, t13_tr], axis=1)
            te_te = np.concatenate([te_te, t13_te], axis=1)
        if KNN:
            from knn_features import knn_block_for_fold

            k_tr, k_te, _nk = knn_block_for_fold(ds, tr_idx, seed, fold, _BASE_TAG)
            te_tr_full = np.concatenate([te_tr_full, k_tr], axis=1)
            te_te = np.concatenate([te_te, k_te], axis=1)
        if IMP:
            from imp_features import imp_block_for_fold

            i_tr, i_te, _ni = imp_block_for_fold(ds, tr_idx, seed, fold, _BASE_TAG)
            te_tr_full = np.concatenate([te_tr_full, i_tr], axis=1)
            te_te = np.concatenate([te_te, i_te], axis=1)
        if DP:
            from dp_features import dp_block

            # fold-invariant (label-free, no cross-fitting needed) — see dp_features.py
            d_tr, d_te, _nd = dp_block(ds)
            te_tr_full = np.concatenate([te_tr_full, d_tr], axis=1)
            te_te = np.concatenate([te_te, d_te], axis=1)
        # per-fold mixed matrices (repair-at-fit-only): fold-train rows from the
        # remasked view, val rows plain — X_all.iloc[tr_idx] is then the fit surface
        cat_all, num_all, miss_all = nd.Xcat_tr, nd.Xnum_tr, nd.miss_tr
        if nd_fit is not None:
            cat_all = nd.Xcat_tr.copy()
            cat_all[tr_idx] = nd_fit.Xcat_tr[tr_idx]
            num_all = nd.Xnum_tr.copy()
            num_all[tr_idx] = nd_fit.Xnum_tr[tr_idx]
            miss_all = nd.miss_tr.copy()
            miss_all[tr_idx] = nd_fit.miss_tr[tr_idx]
        # median-impute raw numerics + the ordinal tail of the TE block (train-fold stats)
        num_med = np.nanmedian(num_all[tr_idx], axis=0)
        num_tr = np.where(np.isnan(num_all), num_med, num_all)
        num_te = np.where(np.isnan(nd.Xnum_te), num_med, nd.Xnum_te)
        te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
        te_tr_f = np.where(np.isnan(te_tr_full), te_med, te_tr_full)
        te_te_f = np.where(np.isnan(te_te), te_med, te_te)

        X_all = frame(cat_all, num_tr, miss_all, te_tr_f)
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

    # ckpt name carries the SURFACE tag (Day-10 audit: r/u runs of the same
    # model+seed must never share checkpoint files after a crash) and the
    # fold-count segment (Day-15: K!=5 folds must never resume 5-fold ckpts)
    ckpt = f"{MODEL_NAME}_{SURFACE_TAG}{_kseg()}_s{SEEDS[0]}"
    oof, test, fold_scores = zoo_cv(ds, fit_fold, ckpt_name=ckpt, seed=SEEDS[0])
    finalize(MODEL_NAME, ds, oof, test, fold_scores)
    clear_ckpt(ckpt)


if __name__ == "__main__":
    main()
