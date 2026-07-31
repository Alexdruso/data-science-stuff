"""Day-23 (evening): a GENERIC M-slope screener — re-adjudicate the rejected inventory.

WHY THIS EXISTS. Day-22 proved that per-seed significance cannot distinguish information
from variance reduction: a variance reducer is significantly positive at EVERY member by
construction and worth nothing at M=8. But the campaign's whole rejected inventory
(`basin`, `focal`, K=10/15, `smooth5`, `te3knn`, `te3nb`, `te3ncw`, `te3p`, `te3rp`,
`te3tri`, `snap3`, `bll`) was screened by exactly that standard — per-seed level reads
against a fixed gate. Day-22 declined to re-screen them for ONE stated reason: they are
single-seed on disk and building seeds "is not affordable".

That was a BUDGET argument, not an epistemic one, and it expires the moment the GPU/CPU
is idle. This script is the tool for spending that budget: give it a challenger key
pattern and a stock key pattern plus the seeds, and it returns the exponent on the same
honest per-arm decision layer used for te13 (+0.558), imp (-0.850) and ji (-1.107).

Usage:
  python src/probe_exponent_any.py NAME "oof_<arm>_s{seed}" "oof_<stock>_s{seed}" 42,7,123

The `{seed}` placeholder is substituted per seed. Writes results/probe_exp_<NAME>.txt.

VERDICT SCALE (Day-22, stated by distance from the two ROBUST anchors — the C2 band is
instrument-specific per Phase 1b and is secondary):
    b > 0     information  (te13 = +0.558; averaging REVEALS it)
    b ~ -0.5  indistinguishable from selection noise
    b ~ -1    variance reduction (C3 = -1.002; dead at M>=8)
    undefined delta crosses zero -> not evidence of information
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

N_SPLITS = 6
ANCHORS = {"te13": +0.558, "C1": -0.027, "C2mid": -0.417, "imp": -0.850, "C3": -1.002}


def run(name: str, arm_pat: str, stock_pat: str, seeds: tuple[int, ...]) -> str:
    y = load_dataset().y

    def stack(pat: str) -> np.ndarray:
        paths = [RESULTS_DIR / f"{pat.format(seed=s)}.npy" for s in seeds]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing arrays for {name}: {missing}")
        return np.stack([np.load(p) for p in paths])

    C, S = stack(arm_pat), stack(stock_pat)
    assert C.shape == S.shape, f"shape mismatch {C.shape} vs {S.shape}"

    cells = []
    for seed in range(N_SPLITS):
        a, b = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(y, y)
        )
        cells += [(a, b), (b, a)]

    res = np.asarray(
        Parallel(n_jobs=6)(delayed(_cell)(C, S, y, f, e, len(seeds)) for f, e in cells)
    )
    mean = res.mean(0)
    ms = np.arange(1, len(seeds) + 1, dtype=float)
    exp = _exponent(ms, mean)
    rng = np.random.default_rng(20260731)
    boot = np.asarray(
        [_exponent(ms, res[rng.integers(0, len(res), len(res))].mean(0)) for _ in range(400)]
    )
    boot = boot[np.isfinite(boot)]
    sd = float(boot.std()) if len(boot) else float("nan")

    lines = [
        "=" * 78,
        f"probe_exponent_any -- {name}",
        f"  arm   {arm_pat}",
        f"  stock {stock_pat}",
        f"  seeds {seeds}, {len(cells)} cells, all subsets per size, honest per-arm dw",
        f"  anchors: " + "  ".join(f"{k} {v:+.3f}" for k, v in ANCHORS.items()),
        "=" * 78,
        "   M       " + "".join(f"    {int(m)}     " for m in ms),
        "   delta   " + "".join(f"  {v:+.6f}" for v in mean),
        "   se      " + "".join(f"  {v:+.6f}" for v in res.std(0) / np.sqrt(len(res))),
        f"   EXPONENT {exp:+.3f}  (bootstrap sd {sd:.3f}, {len(boot)}/400 finite)",
    ]
    if np.isnan(exp):
        v = "UNDEFINED (delta crosses zero) -- not evidence of information"
    elif exp > 0:
        v = "ABOVE ZERO -- the information end of the scale. Escalate and read the LEVEL."
    elif exp > NULL_HI:
        v = "between the null band and zero -- INCONCLUSIVE"
    elif exp > NULL_LO:
        v = "INSIDE the null band -- indistinguishable from selection noise"
    else:
        v = "AT/BELOW C3 -- variance reduction; dies at the deployed breadth"
    lines.append(f"   VERDICT: {v}")
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / f"probe_exp_{name}.txt").write_text(report + "\n")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3], tuple(int(s) for s in sys.argv[4].split(",")))
