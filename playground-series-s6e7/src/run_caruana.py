"""Caruana greedy ensemble selection over ALL repaired (m100) legs on disk.

Question: does any formally-vetoed leg (ftt/rf/mlp/FE variants) earn a seat when
selection is greedy-with-replacement per step instead of one-shot NM? Bases are
precorrected to their deployed decision surfaces first (the s6e7 blend
convention), so the default argmax-on-summed-probs score is decision-aware.
Honest read = `holdout_scores` (outer CV); `full_score` is optimistic.

Run: S6E7_REPAIR=1 python src/run_caruana.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import RESULTS_DIR, precorrect
from train_common import load_dataset

from data_science_stuff.kaggle.stacking import caruana_select

POOL = [
    "lgbm_r_breadth",
    "hgbc_r_breadth",
    "xgboost_r_breadth",
    "catboost_r_breadth",
    "realmlp_r_breadth",
    "ftt_r_s42",
    "rf_r_s42",
    "mlp_r",
    "mlp_la_r",
    "lgbm_te_r_s42",
    "lgbm_freq_r_s42",
    "lgbm_dp_r_s42",
]


def main() -> None:
    ds = load_dataset()
    keys = [k for k in POOL if (RESULTS_DIR / f"oof_{k}.npy").exists()]
    missing = set(POOL) - set(keys)
    if missing:
        print(f"skipping missing: {sorted(missing)}")
    oof_d = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test_d = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    oof_pc, _ = precorrect(oof_d, test_d)
    stacked = np.stack([oof_pc[k] for k in keys])
    res = caruana_select(stacked, ds.y, n_steps=50, init_topk=3, n_outer=5, seed=42)
    print(f"pool ({len(keys)}): {keys}")
    print(f"honest holdout scores: {[round(s, 5) for s in res.holdout_scores]}")
    print(f"mean holdout: {np.mean(res.holdout_scores):.5f}")
    counts = np.bincount(res.full_picks, minlength=len(keys))
    print("full-data pick shares (optimistic, composition only):")
    for k, c in sorted(zip(keys, counts), key=lambda t: -t[1]):
        if c:
            print(f"  {k:<22} {c:>3} ({c / len(res.full_picks):.0%})")


if __name__ == "__main__":
    main()
