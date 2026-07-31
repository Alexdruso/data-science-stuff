"""Day-23 (evening): does a NON-DILUTING leg improvement pass through to the stack?

This is a sharp, falsifiable test of veto #18's stated MECHANISM, and it could not be run
before tonight because the 8-seed basin breadth did not exist until 19:04.

VETO #18 (07-26, `probe_passthrough.py`) measured leg-delta -> stack-delta pass-through at
**2.6% of nominal** and generalised it to "the combiner absorbs leg REPLACEMENTS, so the
whole 'improve a base leg' class has no route to the deliverable." Its stated mechanism:

    "the LR meta is itself an averaging/variance-reduction operator, so leg-level changes
     that work BY AVERAGING or by small mean shifts are exactly what it already extracts."

That mechanism makes a prediction. Tonight's M-slope read says the basin is **NOT** a
variance reducer: exponent +0.475 +-0.291, i.e. 5.1 sd above the C3 anchor, with the delta
GROWING across M=1..3. If veto #18's mechanism is right, a lever that does *not* work by
averaging should NOT be absorbed -- it should pass through. If it is absorbed anyway, then
veto #18 is true but its mechanism is wrong, and the real story is stronger and simpler
(the combiner absorbs leg changes regardless of kind).

⚠️ WHAT VETO #18 ACTUALLY TESTED, and why this is not a repeat. It swapped the basin arm in
at **ONE seed** against a one-seed stock arm (matched, so the read was fair) -- but it could
therefore only ever see the M=1 slice of a lever whose whole claim to being real is what
happens as M grows. This probe replaces the deployed **8-seed** `hgbc_te3_u_breadth` leg
with an **8-seed basin breadth**: matched at the DEPLOYED M, which is the number that
governs the deliverable.

⚠️ PRE-REGISTERED, bars fixed before the read (same instrument and gates as veto #18):
  nominal leg delta = +0.000182 (8/8 seeds positive, measured this evening)
  >= +0.00010 pass-through  => veto #18's mechanism SURVIVES and its generalisation is
                               too broad; record the correction.
  <= +0.00003               => absorbed anyway => veto #18 holds but its MECHANISM is
                               refuted; the absorption is not about averaging.
  between                   => INCONCLUSIVE. Report, do not adopt.

⚠️ THIS CANNOT REACH THE DELIVERABLE AND IS NOT PITCHED AS ONE. Even at 100% pass-through
the level is +0.0002, ~5x under the +0.001 bar, and a #1 swap would need a stack rebuild, a
submission slot, an audit and a user re-click against a same-day deadline. This is run
because the mechanism question is decision-relevant for the NEXT competition, not this one.

Run: python src/probe_passthrough_basin8.py -> results/probe_passthrough_basin8.txt
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

SEEDS = (42, 7, 123, 2024, 99, 2025, 31, 7777)
SWAP = "hgbc_te3_u_breadth"
OUT = RESULTS_DIR / "probe_passthrough_basin8.txt"


def main() -> None:
    y = load_dataset().y
    basin8 = np.mean(
        [np.load(RESULTS_DIR / f"oof_hgbc_te3_u_s{s}t.npy") for s in SEEDS], axis=0
    )
    stock8 = np.load(RESULTS_DIR / f"oof_{SWAP}.npy")
    assert basin8.shape == stock8.shape

    def cols(k: str) -> np.ndarray:
        return clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)

    X_stock = np.hstack([cols(k) for k in KEYS8])
    X_basin = np.hstack(
        [clipped_logit(basin8).astype(np.float32) if k == SWAP else cols(k) for k in KEYS8]
    )
    assert X_stock.shape == X_basin.shape

    cells = []
    for seed in range(N_SPLITS):
        a, b = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(y, y)
        )
        cells += [(a, b), (b, a)]

    s_stock = np.asarray(Parallel(n_jobs=6)(delayed(_arm)(X_stock, y, f, e) for f, e in cells))
    s_basin = np.asarray(Parallel(n_jobs=6)(delayed(_arm)(X_basin, y, f, e) for f, e in cells))
    d = s_basin - s_stock
    nominal = 0.000182
    pt = d.mean() / nominal * 100

    lines = [
        "=" * 78,
        "probe_passthrough_basin8 -- 8-seed basin breadth REPLACES the 8-seed hgbc leg",
        f"  {len(cells)} cells; nominal leg delta +{nominal:.6f} (8/8 seeds positive)",
        "  basin M-slope exponent +0.475 +-0.291 => NOT a variance reducer (5.1 sd above C3)",
        "  bars: >=+0.00010 mechanism survives | <=+0.00003 mechanism refuted | else inconclusive",
        "=" * 78,
        f"  lr_stock   {s_stock.mean():.6f}   (deployed 8-leg set)",
        f"  lr_basin8  {s_basin.mean():.6f}",
        f"  DELTA      {d.mean():+.6f}   se {d.std() / np.sqrt(len(d)):.6f}   "
        f"{int((d > 0).sum())}/{len(d)}   pass-through {pt:.1f}%",
    ]
    if d.mean() >= 0.00010:
        v = ("veto #18's MECHANISM SURVIVES -- a non-averaging lever is NOT absorbed. "
             "Its generalisation to the whole 'improve a leg' class is TOO BROAD.")
    elif d.mean() <= 0.00003:
        v = ("ABSORBED ANYWAY -- veto #18 holds but its stated mechanism ('the meta is an "
             "averaging operator') is REFUTED: absorption is not about averaging.")
    else:
        v = "INCONCLUSIVE between the pre-registered bars. Report, do not adopt."
    lines += ["", f"  VERDICT: {v}",
              "  NOTE: level is ~5x under the +0.001 deliverable bar either way; finals unchanged."]
    report = "\n".join(lines)
    print(report, flush=True)
    OUT.write_text(report + "\n")


if __name__ == "__main__":
    main()
