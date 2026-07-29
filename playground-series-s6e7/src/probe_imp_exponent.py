"""Day-22: the M-slope of `imp` alone — decomposing `ji` into information vs config.

`ji` = joint-TPE config basin (a CONFIG change) x `imp` (the Day-17 driver-reconstruction
INFORMATION channel). Day-21 read `ji`'s trajectory as diluting and stopped there. Today's
validated instrument says the two components should behave differently if the mechanism
story is right: an information lever has exponent ~0, a config/noise lever does not.

`imp` alone had only seed 42 on disk. `run_imp_breadth.sh` added 7 and 123 (2024 was
deliberately not run — 3 seeds is the minimum for an exponent and matches te13's 3-point
fit; stated, not silently truncated).

Instrument: the SAME honest per-arm decision layer as `probe_mslope_te13_honest.py`
(weights fitted on the fit half, scored on the eval half, 12 cells, all subsets per size).
`imp` vs stock is a two-matrix comparison so frozen-w would inherit the calibration
confound documented in Phase 2b.

⚠️ SCOPE, fixed before the read: `imp`'s solo ceiling is ~+0.0002. This is a scientific
data point for screening rule #3' — the first NON-SYNTHETIC candidate for a non-diluting
lever — and is **NOT** a finals candidate. A flat exponent must not be re-read as a route
to the #2 slot; the deliverable bar is +0.001 at the deployed M.

Run: python src/probe_imp_exponent.py -> results/probe_imp_exponent.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from probe_mslope_te13_honest import NULL_HI, NULL_LO, _cell, _exponent
from train_common import RESULTS_DIR, load_dataset

SEEDS = (42, 7, 123)
N_SPLITS = 6
OUT = RESULTS_DIR / "probe_imp_exponent.txt"


def main() -> None:
    y = load_dataset().y
    C = np.stack([np.load(RESULTS_DIR / f"oof_realmlp_te13_imp_u_s{s}.npy") for s in SEEDS])
    S = np.stack([np.load(RESULTS_DIR / f"oof_realmlp_te13_u_s{s}.npy") for s in SEEDS])

    cells = []
    for seed in range(N_SPLITS):
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                           random_state=seed).split(y, y))
        cells += [(a, b), (b, a)]

    res = np.asarray(Parallel(n_jobs=6)(
        delayed(_cell)(C, S, y, f, e, len(SEEDS)) for f, e in cells
    ))
    mean = res.mean(0)
    ms = np.arange(1, len(SEEDS) + 1, dtype=float)
    exp = _exponent(ms, mean)
    rng = np.random.default_rng(20260729)
    b = np.asarray([_exponent(ms, res[rng.integers(0, len(res), len(res))].mean(0))
                    for _ in range(400)])
    b = b[np.isfinite(b)]

    lines = [
        "=" * 78,
        "probe_imp_exponent -- `imp` (driver reconstruction) alone, honest per-arm dw",
        f"seeds {SEEDS}, {len(cells)} cells, all subsets per size",
        f"C2 null band: [{NULL_LO:+.3f}, {NULL_HI:+.3f}]   C1 (mean shift) reference: -0.027",
        "=" * 78,
        "   M       " + "".join(f"    {int(m)}     " for m in ms),
        "   delta   " + "".join(f"  {v:+.6f}" for v in mean),
        "   se      " + "".join(f"  {v:+.6f}" for v in res.std(0) / np.sqrt(len(res))),
    ]
    if mean[0] != 0:
        lines.append("   norm.   " + "".join(f"  {v / mean[0]:+.6f}" for v in mean))
    lines.append(f"   EXPONENT {exp:+.3f}  (bootstrap sd "
                 f"{b.std() if len(b) else float('nan'):.3f}, {len(b)}/400 finite)")
    if np.isnan(exp):
        v = "UNDEFINED (delta crosses zero)"
    elif exp > NULL_HI:
        v = ("MEAN SHIFT -- `imp` is INFORMATION and does not dilute. First non-synthetic "
             "instance. Still ~5x under the deliverable bar; NOT a finals candidate.")
    elif exp < NULL_LO:
        v = "VARIANCE REDUCTION -- dies at the deployed breadth"
    else:
        v = "INSIDE THE NULL BAND -- indistinguishable from selection noise"
    lines.append(f"   VERDICT: {v}")

    report = "\n".join(lines)
    print(report, flush=True)
    OUT.write_text(report + "\n")


if __name__ == "__main__":
    main()
