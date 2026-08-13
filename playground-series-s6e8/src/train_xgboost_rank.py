"""Pairwise-objective XGBoost leg — training on the metric instead of on calibration.

    python src/train_xgboost_rank.py --arm s1 --fold1     # spike; saves nothing
    python src/train_xgboost_rank.py --arm s1 --seeds 42,43,44

**The mechanism.** ROC AUC is `P(score(pos) > score(neg))`. `binary:logistic` optimises
logloss, which is a *proper scoring rule* — it buys calibration, and calibration is
exactly the thing AUC is invariant to. `rank:pairwise` optimises `-log sigmoid(s_i - s_j)`
over sampled positive/negative pairs, which is the BPR/RankNet surrogate for the metric
itself. The two optima genuinely differ.

**Why this lever and not another member.** The Day-10 read found that combining four
families added +0.000024 over `bag(xgb×6)` alone — a strong combiner absorbs added
members (root CLAUDE.md lesson #7). A different *objective* replaces the leg rather than
adding one, and its gain is not averaging-shaped, so it is the class of lever the
combiner cannot manufacture for itself.

**Two implementation traps:**

1. `lambdarank_pair_method` defaults to `"topk"`, which concentrates pairs at the head of
   the ranking — that is the NDCG@k shape. AUC weights *every* pair equally, so this uses
   `"mean"` (uniform pair sampling). Getting this wrong optimises the wrong metric.
2. `predict` returns an **unbounded raw score**, not a probability. `ensemble.py` blends
   in prob/logit/rank space and `bag_seeds` takes a plain mean of probabilities — a raw
   score breaks the first two and makes the third arbitrary. The saved array is therefore
   `expit(score)`: strictly monotone, so the AUC is bit-identical, and every downstream
   space stays well-defined. Note honestly that this is *not* a calibrated probability,
   so **rank space is the honest blend space for this leg**.

One group (`qid` constant) means the per-group AUC XGBoost reports is the global AUC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

sys.path.insert(0, str(Path(__file__).parent))
from legs import N_FOLDS, run_leg
from run_arm import ARMS, assert_budget_constraint, build_matrices
from train_xgboost import PARAMS as LOGISTIC_PARAMS

# Deliberately the *same* tree hyperparameters as the untuned `xgb` leg. The only
# thing this arm changes is the objective — anything else and the paired read stops
# measuring the objective and starts measuring a confound.
PARAMS: dict[str, object] = {
    k: v
    for k, v in LOGISTIC_PARAMS.items()
    if k not in {"objective", "eval_metric", "verbosity"}
} | {
    "objective": "rank:pairwise",
    "lambdarank_pair_method": "mean",  # uniform pairs = AUC-shaped, not NDCG@k-shaped
    "lambdarank_num_pair_per_sample": 8,
    "eval_metric": "auc",
    "verbosity": 0,
}


def make_fit(seed: int, params: dict[str, object]):  # noqa: ANN201 - closure
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = XGBRanker(**{**params, "random_state": seed})
        # A single group: every row is comparable to every other, which is what AUC
        # measures. qid must be non-decreasing, and a constant trivially is.
        model.fit(
            x_tr,
            y_tr,
            qid=np.zeros(len(x_tr), dtype=np.int32),
            eval_set=[(x_va, y_va)],
            eval_qid=[np.zeros(len(x_va), dtype=np.int32)],
            verbose=False,
        )
        # expit: monotone, so AUC is unchanged, but the array lands in (0,1) and every
        # blend space downstream stays defined. Not a calibrated probability.
        return expit(model.predict(x_va)), expit(model.predict(x_te))

    return fit_model


def smoke_fold1(arm: str, seed: int, params: dict[str, object]) -> float:
    """Fit fold 1 only and report its AUC. The spike gate; saves nothing."""
    builder, transform, _ = ARMS[arm]
    x, y, x_test = build_matrices(builder)
    assert_budget_constraint(x)

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(iter(splitter.split(np.zeros(len(x)), y)))
    x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    x_te = x_test
    if transform is not None:
        x_tr, x_va, x_te = transform(x_tr, y_tr, x_va, x_te)

    val, _ = make_fit(seed, params)(x_tr, y_tr, x_va, y_va, x_te, 1)
    auc = float(roc_auc_score(y_va, val))
    print(f"  xgbp_{arm}_s{seed} FOLD 1 ONLY: AUC {auc:.5f}", flush=True)
    return auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--num-pair", type=int, default=None)
    parser.add_argument("--pair-method", default=None, choices=["mean", "topk"])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--fold1", action="store_true")
    args = parser.parse_args()

    params = dict(PARAMS)
    if args.num_pair is not None:
        params["lambdarank_num_pair_per_sample"] = args.num_pair
    if args.pair_method is not None:
        params["lambdarank_pair_method"] = args.pair_method
    if args.n_jobs is not None:
        params["n_jobs"] = args.n_jobs

    print(f"xgbp on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    print(
        f"objective={params['objective']} "
        f"pair_method={params['lambdarank_pair_method']} "
        f"num_pair={params['lambdarank_num_pair_per_sample']}",
        flush=True,
    )
    for seed in (int(s) for s in args.seeds.split(",")):
        if args.fold1:
            smoke_fold1(args.arm, seed, params)
        else:
            run_leg("xgbp", args.arm, seed, make_fit(seed, params))


if __name__ == "__main__":
    main()
