"""Day-22 pre-registration #5, Phase 2: apply the VALIDATED M-slope instrument to the
real levers that have enough seeds on disk.

`probe_mslope.py` calibrated the exponent scale with magnitude-matched synthetic
controls and PASSED its pre-registered validation gate:

    C1 uniform mean shift (real information)  exponent -0.03 +/- 0.01   -> survives averaging
    C2 selection noise (seed half vs half)    exponent -0.42 +/- 0.18   -> the NULL BAND
    C3 variance reduction (identical mean)    exponent -1.00 +/- 0.09   -> dies at 1/M

This probe reads the exponent of levers that actually exist, on the same instrument
(paired arms, decision weights frozen on the all-8 stock breadth, deltas averaged over
random seed orderings, exponent bootstrapped over orderings).

Levers with >= 3 matched seeds on disk -- the whole affordable list:

  ji     realmlp_te13_imp_u_s*ji  vs realmlp_te13_u_s*   (4 seeds)  Day-21's rule-#3 exemplar
  te13   realmlp_te13_u_s*        vs realmlp_u_s*        (3 seeds)  a KNOWN-REAL recipe lever
                                                                    (+0.0004, the 07-11 find)

STATED INVENTORY LIMIT, no silent truncation: `te3knn`, `te3nb`, `te3ncw`, `te3p`,
`te3rp`, `te3tri`, `s42t` (basin), `snap3`, `bll`, `smooth5` and the rest of the rejected
micro-positive ledger are SINGLE-SEED on disk. An exponent needs >= 3 members per arm, so
they cannot be re-adjudicated without rebuilding seeds, and they are not read here.

Run:  python src/probe_mslope_phase2.py  ->  results/probe_mslope_phase2.txt
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

RNG_SEED = 20260729
OUT = RESULTS_DIR / "probe_mslope_phase2.txt"

# (label, challenger keys, stock keys) -- matched seed order, paired arms
LEVERS: list[tuple[str, list[str], list[str]]] = [
    (
        "ji (joint basin x IMP)",
        [f"realmlp_te13_imp_u_s{s}ji" for s in (42, 7, 123, 2024)],
        [f"realmlp_te13_u_s{s}" for s in (42, 7, 123, 2024)],
    ),
    (
        "te13 recipe (KNOWN REAL)",
        [f"realmlp_te13_u_s{s}" for s in (42, 7, 123)],
        [f"realmlp_u_s{s}" for s in (42, 7, 123)],
    ),
]


def _ba(y: NDArray[np.int64], p: NDArray[np.float32], w: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, np.argmax(p * w, axis=1)))


def _exponent(ms: NDArray[np.float64], d: NDArray[np.float64]) -> float:
    if np.any(d <= 0):
        return float("nan")
    return float(np.polyfit(np.log(ms), np.log(d), 1)[0])


def main() -> None:
    ds = load_dataset()
    y = ds.y
    rng = np.random.default_rng(RNG_SEED)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("probe_mslope_phase2 -- the validated instrument applied to real levers")
    say("=" * 78)
    say("calibration from probe_mslope.py (leg realmlp_te13_u, 60 reps):")
    say("   C1 mean shift        exponent -0.027 +/- 0.012   [real information]")
    say("   C2 selection noise   exponent -0.417 +/- 0.178   [NULL BAND -0.773 .. -0.060]")
    say("   C3 variance reduc.   exponent -1.002 +/- 0.086   [dies at 1/M]")
    say()
    null_lo, null_hi = -0.773, -0.060

    for label, chal_keys, stock_keys in LEVERS:
        C = np.stack([np.load(RESULTS_DIR / f"oof_{k}.npy").astype(np.float32) for k in chal_keys])
        S = np.stack([np.load(RESULTS_DIR / f"oof_{k}.npy").astype(np.float32) for k in stock_keys])
        m_max = len(C)
        # Decision weights FROZEN on the stock arm's full breadth -- one w for both arms,
        # every M, so nothing in the read is a re-tuning of w.
        wcache = RESULTS_DIR / f"mslope_w_p2_{chal_keys[0]}.json"
        if wcache.exists():
            w = np.asarray(json.loads(wcache.read_text()), dtype=np.float64)
        else:
            w = robust_decision_weights(y, S.mean(axis=0).astype(np.float64))
            wcache.write_text(json.dumps(w.tolist()))

        orders = list(itertools.permutations(range(m_max)))
        if len(orders) > 24:
            orders = [tuple(rng.permutation(m_max)) for _ in range(24)]
        traj = []
        for o in orders:
            row = []
            for m in range(1, m_max + 1):
                idx = list(o[:m])
                row.append(_ba(y, C[idx].mean(0), w) - _ba(y, S[idx].mean(0), w))
            traj.append(row)
        A = np.asarray(traj)
        mean = A.mean(axis=0)
        ms = np.arange(1, m_max + 1, dtype=np.float64)
        exp = _exponent(ms, mean)
        boots = []
        for _ in range(400):
            idx = rng.integers(0, A.shape[0], A.shape[0])
            boots.append(_exponent(ms, A[idx].mean(axis=0)))
        b = np.asarray(boots)
        b = b[np.isfinite(b)]
        sd = float(b.std()) if len(b) else float("nan")

        say(f"-- {label}   ({m_max} matched seeds, {len(orders)} orderings) " + "-" * 8)
        say(f"   stock breadth-{m_max} dw-BA = {_ba(y, S.mean(0), w):.6f}   "
            f"challenger = {_ba(y, C.mean(0), w):.6f}")
        say("   M            " + "".join(f"    {int(m)}     " for m in ms))
        say("   delta        " + "".join(f"  {v:+.6f}" for v in mean))
        say("   normalised   " + "".join(f"  {v / mean[0]:+.6f}" for v in mean))
        say(f"   EXPONENT (M=1..{m_max}):  {exp:+.3f}  (bootstrap sd {sd:.3f})")
        if np.isnan(exp):
            verdict = "UNDEFINED (delta crosses zero) -- consistent with noise"
        elif exp > null_hi:
            verdict = "MEAN SHIFT -- survives averaging, escalate at full size"
        elif exp < null_lo:
            verdict = "VARIANCE REDUCTION -- worth ~1/M, dead at the deployed breadth"
        else:
            verdict = "INSIDE THE NULL BAND -- indistinguishable from selection noise"
        say(f"   VERDICT: {verdict}")
        say()

    say("-- published trajectories re-fitted over M=1..4 for comparability -----------")
    for name, d in (
        ("snap4 (honest A/B, Day-21)", [3.31e-4, 2.14e-4, 1.40e-4, 1.17e-4]),
        ("snap3 (honest A/B, Day-21)", [2.31e-4, 1.72e-4, 0.84e-4, 0.70e-4]),
        ("snap2 (honest A/B, Day-21)", [1.76e-4, 1.06e-4, 0.31e-4, 0.24e-4]),
        ("ji (Day-21 single ordering)", [2.06e-4, 1.38e-4, 0.89e-4, 0.80e-4]),
    ):
        e = _exponent(np.arange(1.0, 5.0), np.asarray(d))
        tag = ("mean shift" if e > null_hi else "variance reduction" if e < null_lo else "null band")
        say(f"   {name:<30} exponent {e:+.3f}   -> {tag}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
