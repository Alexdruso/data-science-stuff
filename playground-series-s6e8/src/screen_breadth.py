"""M-slope screen for the breadth delta (the hold-band action).

    python src/screen_breadth.py

The Day-6 pre-registration puts a blend delta in +0.0003..+0.0010 into the
**hold** band, whose prescribed action is "re-screen before adopting". This is
that screen, in the shape root CLAUDE.md lesson #1 defines: measure the paired
delta at M = 1, 2, ... members per family and read how it scales.

    b ~  0   information      — averaging reveals it; survives at deployed breadth
    b ~ -0.5 selection noise  — the winner's curse
    b ~ -1   variance reduction — dead at any deployed breadth

**Weights are fixed, not refit per subset.** Refitting Nelder-Mead inside every
(lgbm-subset, xgb-subset) pairing would both cost ~80 optimisations over 691k
rows and, worse, inject exactly the selection noise the screen is trying to
detect — each subset would get its own winner's-curse weight. A fixed blend
isolates the mechanism. The weight used is the full-data optimum, applied
identically at every M, so nothing is chosen per-point.

Pairings are capped per M so cost stays linear; the cap is printed rather than
silently applied (root CLAUDE.md: no silent caps).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import to_rank
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

MAX_PAIRINGS = 6
W_XGB = 0.78  # full-data Nelder-Mead optimum; held fixed across every M


def main() -> None:
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()

    lgbm = {
        s: np.load(RESULTS_DIR / f"oof_s1_s{s}.npy").ravel()
        for s in range(42, 48)
        if (RESULTS_DIR / f"oof_s1_s{s}.npy").exists()
    }
    xgb = {
        s: np.load(RESULTS_DIR / f"oof_xgb_s1_s{s}.npy").ravel()
        for s in range(42, 48)
        if (RESULTS_DIR / f"oof_xgb_s1_s{s}.npy").exists()
    }
    m_max = min(len(lgbm), len(xgb))
    print(f"lgbm seeds {sorted(lgbm)}, xgb seeds {sorted(xgb)} -> M up to {m_max}")
    print(f"fixed w_xgb={W_XGB}, <= {MAX_PAIRINGS} pairings per M\n")

    deltas = []
    for m in range(1, m_max + 1):
        lsub = list(itertools.combinations(sorted(lgbm), m))[:MAX_PAIRINGS]
        xsub = list(itertools.combinations(sorted(xgb), m))[:MAX_PAIRINGS]
        vals = []
        for lc, xc in zip(lsub, xsub):
            lbag = np.mean([lgbm[s] for s in lc], axis=0)
            xbag = np.mean([xgb[s] for s in xc], axis=0)
            null = float(roc_auc_score(y, lbag))
            mixed = (1 - W_XGB) * to_rank(lbag) + W_XGB * to_rank(xbag)
            vals.append(float(roc_auc_score(y, mixed)) - null)
        deltas.append(float(np.mean(vals)))
        print(
            f"  M={m}  delta={np.mean(vals):+.6f}  sd={np.std(vals):.6f}  "
            f"(n={len(vals)})",
            flush=True,
        )

    if len(deltas) >= 4:
        ms = np.arange(1, len(deltas) + 1)
        b = float(np.polyfit(np.log(ms), np.log(np.abs(deltas)), 1)[0])
        print(f"\n  fitted exponent b = {b:+.3f}  over M=1..{len(deltas)}")
        print("  [~0 information | ~-0.5 selection noise | ~-1 variance reduction]")
    else:
        print(f"\n  only {len(deltas)} points — lesson #4: do not fit an exponent here.")
        print("  Read the direction only; refit once six xgb seeds exist.")


if __name__ == "__main__":
    main()
