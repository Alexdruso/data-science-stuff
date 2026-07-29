"""Day-22 pre-registration #5: is the **M-slope** an INSTRUMENT, or just a heuristic?

See CLAUDE.md "PRE-REGISTRATION #5" for the full design and the bars. Short version:

Day-21's screening rule #3 ("a per-member gain is worth ~1/M at the deployed breadth")
was induced from TWO levers (`snap4`, `ji`), both of which decayed. It was never tested
against a lever KNOWN to be real, and it has no null distribution -- so "ji decays at
M^-0.71" cannot currently be distinguished from "any 1e-4-scale reading decays like
that". This probe supplies the missing controls, all MAGNITUDE-MATCHED to ~+0.0003 at
M=1 (the size of snap4/ji), all from per-seed OOF arrays already on disk, zero GPU.

Mechanistic predictions under test:

    C1  uniform MEAN SHIFT (real information)   -> exponent ~  0    (survives averaging)
    C2  SELECTION NOISE (seed-half vs seed-half) -> exponent ~ -0.5  (CLT on a difference)
    C3  VARIANCE REDUCTION (shrink to a disjoint
        member average; identical mean)          -> exponent ~ -1    (competes with averaging)

If those separate, the exponent tells you WHAT KIND of thing you measured before any GPU
is spent -- and, uncomfortably, it says that "significant at 4/4 seeds, se 2.2e-5" (the
evidence `ji` presented) is the *signature of a variance reducer*, since a variance
reducer is positive at every individual member and worth nothing at M=8.

Protocol
  * Base = a real leg's 8 per-seed OOF arrays. Each replicate randomly splits the 8 seeds
    into an ARM half (4) and a disjoint TARGET half (4), so no arm is ever shrunk toward
    a member it contains (that would force delta=0 at full M by construction).
  * Every arm is PAIRED: challenger and stock average the SAME M members.
  * Metric = the deployed one: balanced accuracy under decision weights fitted ONCE per
    leg on the all-8 breadth and then FROZEN, so no read is contaminated by re-tuning w.
    Plain argmax reported as a secondary.
  * lambda for C1/C3 calibrated by bisection so both read ~+0.0003 at M=1.
  * Exponent = OLS of log|delta| on log M over M=1..4; uncertainty = bootstrap over
    replicates. C2's bootstrap distribution IS the null band.

C1 consumes the labels and is QUARANTINED instrument calibration -- it can never become a
lever. C1b (shrink toward the deployed stack posterior) is the label-free corroboration.

Run:  python src/probe_mslope.py            -> results/probe_mslope.txt
Env:  S6E7_MSLOPE_LEG   (default realmlp_te13_u; the deployed #2's leg)
      S6E7_MSLOPE_REPS  (default 60)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777]
LEG = os.environ.get("S6E7_MSLOPE_LEG", "realmlp_te13_u")
N_REPS = int(os.environ.get("S6E7_MSLOPE_REPS", "60"))
M_MAX = 4  # arm half size; also the range ji is available over
TARGET_DELTA = 3.0e-4  # magnitude match to snap4 (+0.00033) / ji (+0.00025)
RNG_SEED = 20260729
OUT = RESULTS_DIR / "probe_mslope.txt"


def _ba(y: NDArray[np.int64], p: NDArray[np.float32], w: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, np.argmax(p * w, axis=1)))


def _frozen_weights(leg: str, y: NDArray[np.int64], stack: NDArray[np.float32]) -> NDArray[np.float64]:
    """Decision weights fitted once on the all-8 breadth, cached, then never refitted."""
    cache = RESULTS_DIR / f"mslope_w_{leg}.json"
    if cache.exists():
        return np.asarray(json.loads(cache.read_text()), dtype=np.float64)
    w = robust_decision_weights(y, stack.astype(np.float64))
    cache.write_text(json.dumps(w.tolist()))
    return w


def _load_leg(leg: str) -> NDArray[np.float32]:
    arrs = []
    for s in SEEDS:
        p = RESULTS_DIR / f"oof_{leg}_s{s}.npy"
        assert p.exists(), f"missing {p} -- this probe needs all 8 per-seed arrays"
        arrs.append(np.load(p).astype(np.float32))
    return np.stack(arrs)


def _mix(a: NDArray[np.float32], b: NDArray[np.float32], lam: float) -> NDArray[np.float32]:
    return (1.0 - lam) * a + lam * b


def _calibrate(fn, lo: float, hi: float, target: float, n: int = 18) -> float:
    """Bisect lam in [lo, hi] so fn(lam) == target (fn assumed monotone increasing)."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _exponent(ms: NDArray[np.float64], deltas: NDArray[np.float64]) -> float:
    """OLS slope of log|delta| on log M. Returns nan if any delta is non-positive."""
    if np.any(deltas <= 0):
        return float("nan")
    x = np.log(ms)
    yl = np.log(deltas)
    return float(np.polyfit(x, yl, 1)[0])


def main() -> None:
    ds = load_dataset()
    y = ds.y
    P = _load_leg(LEG)  # (8, N, 3)
    n = P.shape[1]
    assert len(y) == n, f"row mismatch: y={len(y)} vs oof={n}"

    breadth = P.mean(axis=0)
    w = _frozen_weights(LEG, y, breadth)
    onehot = np.zeros((n, 3), dtype=np.float32)
    onehot[np.arange(n), y] = 1.0
    stack_path = RESULTS_DIR / "oof_lrstack_u4_r.npy"
    stack = np.load(stack_path).astype(np.float32) if stack_path.exists() else None

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("probe_mslope -- pre-registration #5: is the M-slope an instrument?")
    say("=" * 78)
    say(f"leg={LEG}   seeds={SEEDS}   rows={n}   reps={N_REPS}   M=1..{M_MAX}")
    say(f"frozen decision weights (all-8 breadth): {np.round(w, 5).tolist()}")
    say(f"breadth-8 dw-BA = {_ba(y, breadth, w):.6f}    per-seed dw-BA = "
        f"{np.round([_ba(y, P[i], w) for i in range(8)], 6).tolist()}")
    say()

    # ---- calibrate lambdas on a fixed reference member (seed 42) at M=1 ----------------
    rng = np.random.default_rng(RNG_SEED)
    ref = P[0]
    ref_ba = _ba(y, ref, w)
    tgt_half = P[4:].mean(axis=0)  # disjoint 4-member average, for the C3 shrink target

    lam1 = _calibrate(lambda L: _ba(y, _mix(ref, onehot, L), w) - ref_ba, 0.0, 0.02, TARGET_DELTA)
    lam3 = _calibrate(lambda L: _ba(y, _mix(ref, tgt_half, L), w) - ref_ba, 0.0, 0.5, TARGET_DELTA)
    lam1b = None
    if stack is not None:
        lam1b = _calibrate(lambda L: _ba(y, _mix(ref, stack, L), w) - ref_ba, 0.0, 0.5, TARGET_DELTA)

    say("-- lambda calibration (seed-42 member, M=1, target +%.5f) ------------------" % TARGET_DELTA)
    say(f"  C1  mean shift   -> e_y            lambda = {lam1:.6f}   "
        f"M=1 delta {_ba(y, _mix(ref, onehot, lam1), w) - ref_ba:+.6f}")
    say(f"  C3  var. reduc.  -> disjoint 4-avg lambda = {lam3:.6f}   "
        f"M=1 delta {_ba(y, _mix(ref, tgt_half, lam3), w) - ref_ba:+.6f}")
    if lam1b is not None:
        say(f"  C1b mean shift   -> u4 stack       lambda = {lam1b:.6f}   "
            f"M=1 delta {_ba(y, _mix(ref, stack, lam1b), w) - ref_ba:+.6f}")
    say()

    # ---- replicate loop ---------------------------------------------------------------
    ms = np.arange(1, M_MAX + 1, dtype=np.float64)
    keys = ["C1_mean_shift", "C3_var_reduction", "C2_selection_noise"]
    if lam1b is not None:
        keys.insert(2, "C1b_shift_to_stack")
    traj: dict[str, list[list[float]]] = {k: [] for k in keys}

    for _ in range(N_REPS):
        perm = rng.permutation(8)
        arm, tgt = perm[:4], perm[4:]
        tgt_avg = P[tgt].mean(axis=0)
        # C2: which half is "the challenger" is decided by the M=1 read (winner's curse,
        # exactly as a real screen would do it).
        a1, b1 = _ba(y, P[arm[0]], w), _ba(y, P[tgt[0]], w)
        hi_arm, lo_arm = (arm, tgt) if a1 >= b1 else (tgt, arm)

        rows: dict[str, list[float]] = {k: [] for k in keys}
        for m in range(1, M_MAX + 1):
            sub = P[arm[:m]]
            stock = sub.mean(axis=0)
            s_ba = _ba(y, stock, w)
            rows["C1_mean_shift"].append(_ba(y, _mix(stock, onehot, lam1), w) - s_ba)
            rows["C3_var_reduction"].append(
                _ba(y, _mix(sub, tgt_avg[None], lam3).mean(axis=0), w) - s_ba
            )
            if lam1b is not None:
                rows["C1b_shift_to_stack"].append(_ba(y, _mix(stock, stack, lam1b), w) - s_ba)
            rows["C2_selection_noise"].append(
                _ba(y, P[hi_arm[:m]].mean(axis=0), w) - _ba(y, P[lo_arm[:m]].mean(axis=0), w)
            )
        for k in keys:
            traj[k].append(rows[k])

    # ---- report -----------------------------------------------------------------------
    boot = rng  # reuse
    say("-- mean delta trajectory (dw-BA, averaged over %d replicates) --------------" % N_REPS)
    say("  arm                     " + "".join(f"   M={int(m)}    " for m in ms))
    summary: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        A = np.asarray(traj[k])  # (reps, M)
        mean = A.mean(axis=0)
        say(f"  {k:<22}" + "".join(f"  {v:+.6f}" for v in mean))
        # bootstrap the exponent over replicates
        exps = []
        for _ in range(400):
            idx = boot.integers(0, A.shape[0], A.shape[0])
            exps.append(_exponent(ms, A[idx].mean(axis=0)))
        e = np.asarray(exps, dtype=np.float64)
        e = e[np.isfinite(e)]
        summary[k] = (_exponent(ms, mean), float(e.std()), float(np.mean(mean[0])))
    say()
    say("  normalised to own M=1:")
    for k in keys:
        A = np.asarray(traj[k]).mean(axis=0)
        say(f"  {k:<22}" + "".join(f"  {v / A[0]:+.6f}" for v in A))
    say(f"  {'1/M':<22}" + "".join(f"  {1.0 / m:+.6f}" for m in ms))
    say(f"  {'1/sqrt(M)':<22}" + "".join(f"  {1.0 / np.sqrt(m):+.6f}" for m in ms))
    say()

    say("-- EXPONENTS (OLS of log|delta| on log M, M=1..%d; bootstrap sd) -----------" % M_MAX)
    say("  arm                      exponent   boot sd   predicted")
    pred = {
        "C1_mean_shift": " 0.0 (survives averaging)",
        "C1b_shift_to_stack": " 0.0 (label-free corroboration)",
        "C3_var_reduction": "-1.0 (competes with averaging)",
        "C2_selection_noise": "-0.5 (CLT on a difference)",
    }
    for k in keys:
        exp, sd, _ = summary[k]
        say(f"  {k:<22}   {exp:+.3f}     {sd:.3f}    {pred[k]}")
    say()

    # ---- the pre-registered validation gate -------------------------------------------
    e1, s1, _ = summary["C1_mean_shift"]
    e2, s2, _ = summary["C2_selection_noise"]
    e3, s3, _ = summary["C3_var_reduction"]
    null_hi, null_lo = e2 + 2 * s2, e2 - 2 * s2
    sep = (e1 - e3) / max(np.hypot(s1, s3), 1e-9)
    say("-- PRE-REGISTERED VALIDATION GATE ------------------------------------------")
    say(f"  C2 null band (exponent +/- 2sd): [{null_lo:+.3f}, {null_hi:+.3f}]")
    say(f"  C1 above null band upper edge?   {e1 > null_hi}   ({e1:+.3f} vs {null_hi:+.3f})")
    say(f"  C3 below null band lower edge?   {e3 < null_lo}   ({e3:+.3f} vs {null_lo:+.3f})")
    say(f"  C1 - C3 separation in sd units:  {sep:.1f}  (bar: >= 3)")
    passed = bool(e1 > null_hi and e3 < null_lo and sep >= 3.0)
    say(f"  ==> VALIDATION {'PASS' if passed else 'FAIL'}")
    say()
    say("  reference points already on the record (Day-21):")
    say("    snap4 (honest A/B, M=1..8)  exponent -1.15   -> reads as VARIANCE REDUCTION")
    say("    ji    (paired dw-OOF, M=1..4) exponent -0.71 -> between noise and var. reduction")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
