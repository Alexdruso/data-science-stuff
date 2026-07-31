"""Day-23 (evening): the SEAT test I skipped when I closed `dp` on a solo read.

The morning lane measured `dp` (label-free driver posteriors, 20 cols) SOLO and by
M-slope, and closed it. That was half the evidence: every other candidate in this
campaign got a 12-cell stack read as well, and the stack is the consumer that matters
for final #1. Two consumers have disagreed before — Day-17's reconstruction channel
took seat veto #14 from a TREE canary and had to be re-earned as veto #17 through the
smooth consumer — so "solo null" does not license "seat null".

Zero new training: `dp`'s 3 seeds are on disk from the morning lane.

  lr8   — incumbent 8-leg set
  lrdp  — lr8 + the dp arm's logits (3-seed breadth, matched to the incumbent's
          breadth legs so this is not the single-seed handicap Day-14 F3 refuted)
  lrdp1 — lr8 + the dp arm at ONE seed (s42), as the breadth-vs-single control

⚠️ PRE-REGISTERED BAR, same as every seat test in this campaign: the pooled MDE of a
12-cell probe is 6.5e-5 (`probe_veto_power`), and the adoption gate is +0.0004. Anything
under that is a veto. `dp` is expected to veto — the point is that the record should say
so on MEASURED evidence at both consumers, not on a solo read plus an assumption.

Run: python src/probe_seat_dp.py -> results/probe_seat_dp.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from probe_seat_day17 import KEYS8, N_SPLITS, _arm
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit

SEEDS = (42, 7, 123)
OUT = RESULTS_DIR / "probe_seat_dp.txt"
GATE = 0.0004


def main() -> None:
    y = load_dataset().y
    X8 = np.hstack(
        [clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32) for k in KEYS8]
    )
    per_seed = [np.load(RESULTS_DIR / f"oof_realmlp_te13_dp_u_s{s}.npy") for s in SEEDS]
    breadth = np.mean(per_seed, axis=0)

    mats = {
        "lr8": X8,
        "lrdp": np.hstack([X8, clipped_logit(breadth).astype(np.float32)]),
        "lrdp1": np.hstack([X8, clipped_logit(per_seed[0]).astype(np.float32)]),
    }
    print({n: X.shape for n, X in mats.items()}, flush=True)

    cells = []
    for seed in range(N_SPLITS):
        a, b = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(y, y)
        )
        cells += [(a, b), (b, a)]

    scores = {
        n: np.asarray(Parallel(n_jobs=6)(delayed(_arm)(X, y, f, e) for f, e in cells))
        for n, X in mats.items()
    }

    base = scores["lr8"]
    lines = [
        "=" * 78,
        "probe_seat_dp -- does the driver-POSTERIOR block earn a seat in the 8-leg stack?",
        f"seeds {SEEDS}, {len(cells)} cells; adoption gate +{GATE:.4f}; 12-cell MDE 6.5e-5",
        "=" * 78,
        f"  lr8      {base.mean():.6f}   (incumbent)",
    ]
    for n in ("lrdp", "lrdp1"):
        d = scores[n] - base
        lines.append(
            f"  {n:8s} {scores[n].mean():.6f}   {d.mean():+.6f}   "
            f"se {d.std() / np.sqrt(len(d)):.6f}   {int((d > 0).sum())}/{len(d)}"
        )
    best = max((scores[n] - base).mean() for n in ("lrdp", "lrdp1"))
    lines.append("")
    lines.append(
        f"  VERDICT: {'SEAT' if best >= GATE else 'VETO'} -- best challenger {best:+.6f} "
        f"vs gate +{GATE:.4f}"
    )
    report = "\n".join(lines)
    print(report, flush=True)
    OUT.write_text(report + "\n")


if __name__ == "__main__":
    main()
