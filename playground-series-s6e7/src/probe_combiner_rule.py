"""Focused rule-leg seat test (Day-9 pm): lr6 vs add/swap realmlp_rule_r_breadth.

The Day-9 tournament's lr9 arm bundled realmlp_rule with cascade+tabms (which
dilute); this isolates the one live question after the user's public reads:
does the rule-recipe realmlp earn a seat CLEANLY —
  lr6r = lr6 + realmlp_rule_r_breadth as 7th base (add)
  lr6s = lr6 with realmlp_r_breadth REPLACED by realmlp_rule_r_breadth (swap)
Same 12 split-half cells, same displacement gate vs lr6.

Run: S6E7_REPAIR=1 python src/probe_combiner_rule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from build_lr_stack import DEFAULT_KEYS
from ensemble import RESULTS_DIR, precorrect
from probe_combiner_day9 import DISPLACE_GATE, LR6_EXTRA, cell
from train_common import load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit

N_SPLITS = 6
RULE = "realmlp_rule_r_breadth"


def main() -> None:
    keys = DEFAULT_KEYS
    all_keys = sorted(set(keys + LR6_EXTRA + [RULE]))
    ds = load_dataset()
    y = ds.y
    oof_raw = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in all_keys}
    test_raw = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in all_keys}
    oof_pc_d, _ = precorrect(
        {k: oof_raw[k] for k in keys}, {k: test_raw[k] for k in keys}
    )
    oof_pc = [oof_pc_d[k] for k in keys]
    logit_d = {k: clipped_logit(oof_raw[k]) for k in all_keys}
    base6 = [logit_d[k] for k in keys + LR6_EXTRA]
    swap6 = [
        logit_d[RULE if k == "realmlp_r_breadth" else k] for k in keys + LR6_EXTRA
    ]
    arm_blocks = {
        "lr6": base6,
        "lr6r": base6 + [logit_d[RULE]],
        "lr6s": swap6,
    }

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(oof_pc, arm_blocks, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    print(f"focused rule-seat tournament; {len(cells)} cells; incumbent = lr6")
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr6"]
        print(
            f"  {a:<8} {scores[a].mean():.5f}   vs lr6 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    for a in ("lr6r", "lr6s"):
        d = scores[a] - scores["lr6"]
        gate = d.mean() >= DISPLACE_GATE and (d > 0).sum() >= 10
        print(f"{a}: {'PASSES displacement gate' if gate else 'below gate'}")


if __name__ == "__main__":
    main()
