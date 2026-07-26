"""Day-21 hail Mary: does a LEG upgrade pass through to the STACK?

Day-20 measured that the sub-bar micro-positives partially ADD at the leg level
(~50-60% of nominal).  Every test of a modified leg so far has been an ADDED-leg
seat test (17 vetoes) — the combiner absorbs new columns.  Replacing a leg the
stack demonstrably LOADS ONTO (probe_combiner_u: lr = lr2, the top-2 legs carry
the whole stack) is a different question and has never been measured.

Both upgraded arms already exist on disk, so this costs no training:

  hgbc_te3_u_s42t            TPE basin config        +0.000255 solo (dw OOF)
  realmlp_te13_imp_u_s42ji   joint basin x IMP       +0.000198 solo (dw OOF)
                                     combined nominal +0.000453

Arms (matched N=1 at the two swapped positions; the other 6 legs are identical
across arms, so every read is paired):

  lr_stock     KEYS8 with both breadths demoted to their seed-42 singles
  lr_hg        lr_stock + hgbc position swapped to the basin leg
  lr_rm        lr_stock + realmlp position swapped to the joint x IMP leg
  lr_both      both positions swapped          <-- the headline
  lr_deployed  the real KEYS8 (8-12 seed breadths), level reference only

Pre-registered bars on (lr_both - lr_stock), see CLAUDE.md:
  >= +0.00020  escalate to full breadth parity
  <= +0.00005  pass-through absent, pitch dies
  otherwise    inconclusive, no adoption

Run: python src/probe_passthrough.py -> results/probe_passthrough.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from probe_seat_day17 import KEYS8, _arm
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit

N_SPLITS = 6  # -> 12 cells (both halves of each split)
HGBC_POS = KEYS8.index("hgbc_te3_u_breadth")
RM_POS = KEYS8.index("realmlp_te13_u_breadth")

STOCK = list(KEYS8)
STOCK[HGBC_POS] = "hgbc_te3_u_s42"
STOCK[RM_POS] = "realmlp_te13_u_s42"

HGBC_UP = "hgbc_te3_u_s42t"
RM_UP = "realmlp_te13_imp_u_s42ji"

# solo dw-OOF deltas of the two upgrades, from results/cv_scores.csv (pre-registered)
NOMINAL = {"lr_hg": 0.000255, "lr_rm": 0.000198, "lr_both": 0.000453}


def _legs(keys: list[str]) -> np.ndarray:
    return np.hstack(
        [
            clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
            for k in keys
        ]
    )


def main() -> None:
    y = load_dataset().y

    swap = dict.fromkeys(STOCK)  # keep insertion order for the report
    arms = {
        "lr_stock": list(STOCK),
        "lr_hg": [HGBC_UP if k == STOCK[HGBC_POS] else k for k in STOCK],
        "lr_rm": [RM_UP if k == STOCK[RM_POS] else k for k in STOCK],
        "lr_both": [
            HGBC_UP
            if k == STOCK[HGBC_POS]
            else (RM_UP if k == STOCK[RM_POS] else k)
            for k in STOCK
        ],
        "lr_deployed": list(KEYS8),
    }
    del swap

    mats = {name: _legs(keys) for name, keys in arms.items()}
    for name, X in mats.items():
        print(f"{name:12s} {X.shape}  {arms[name]}", flush=True)

    cells: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    def cell(fit_idx, eval_idx):  # noqa: ANN001, ANN202
        return {name: _arm(X, y, fit_idx, eval_idx) for name, X in mats.items()}

    results = Parallel(n_jobs=4, verbose=5)(
        delayed(cell)(f, e) for f, e in cells
    )
    scores = {a: np.array([r[a] for r in results]) for a in results[0]}

    lines = [
        "PASS-THROUGH PROBE (Day-21 hail Mary) — leg upgrade -> stack delta",
        f"{len(cells)} split-half cells; matched N=1 at the two swapped positions",
        f"stock legs   = {STOCK}",
        f"upgrades     = {HGBC_UP} / {RM_UP}",
        "PRE-REG BARS on lr_both - lr_stock: >=+0.00020 escalate / "
        "<=+0.00005 dies / else inconclusive",
        "",
    ]
    for a in ("lr_stock", "lr_hg", "lr_rm", "lr_both", "lr_deployed"):
        d = scores[a] - scores["lr_stock"]
        extra = ""
        if a in NOMINAL:
            pt = d.mean() / NOMINAL[a] if NOMINAL[a] else float("nan")
            extra = f"   nominal {NOMINAL[a]:+.5f}  pass-through {pt * 100:5.1f}%"
        lines.append(
            f"  {a:<12} {scores[a].mean():.6f}  vs stock {d.mean():+.6f} "
            f"(sd {d.std(ddof=1):.6f}, se {d.std(ddof=1) / np.sqrt(len(d)):.6f}, "
            f"positive {int((d > 0).sum())}/{len(d)}){extra}"
        )

    d = scores["lr_both"] - scores["lr_stock"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    lines += [
        "",
        f"HEADLINE  lr_both - lr_stock = {d.mean():+.6f}  (se {se:.6f}, "
        f"{d.mean() / se:+.2f} sd from 0)",
    ]
    if d.mean() >= 0.00020:
        lines.append("VERDICT: ESCALATE — pass-through clears the pre-registered bar.")
    elif d.mean() <= 0.00005:
        lines.append("VERDICT: DIES — pass-through absent; u4 stays, no GPU spent.")
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no adoption.")

    # additivity of the two swaps AT THE STACK level (free by-product)
    s_hg = (scores["lr_hg"] - scores["lr_stock"]).mean()
    s_rm = (scores["lr_rm"] - scores["lr_stock"]).mean()
    lines.append(
        f"stack-level additivity: hg {s_hg:+.6f} + rm {s_rm:+.6f} = {s_hg + s_rm:+.6f} "
        f"vs both {d.mean():+.6f}  ({d.mean() / (s_hg + s_rm) * 100:.0f}% of the sum)"
        if abs(s_hg + s_rm) > 1e-9
        else "stack-level additivity: parts sum ~0, ratio undefined"
    )

    out = "\n".join(lines)
    print(out)
    (RESULTS_DIR / "probe_passthrough.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
