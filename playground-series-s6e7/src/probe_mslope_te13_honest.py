"""Day-22 follow-up: is `te13`'s FLAT M-slope real, or a frozen-w calibration artefact?

Phase 2 read te13's exponent as +0.146 under decision weights FROZEN on the stock
(non-te13) arm. That instrument is neutral for the C1/C2/C3 controls -- they perturb a
single arm, so both arms share a calibration -- but te13 compares two DIFFERENT feature
matrices, whose posteriors are calibrated differently. That is exactly the confound that
flipped `ji` from +0.000206 (per-arm dw) to -0.000027 (frozen dw).

So this re-reads the same three matched seeds with an HONEST PER-ARM decision layer:
weights fitted on the fit half, scored on the eval half, both directions, 6 splits => 12
cells. Each arm gets its own w (deployment-matched) with no in-sample optimism.

  exponent still above the C2 null band  => te13 genuinely does not dilute; the
                                            "real information survives averaging" claim
                                            keeps its non-synthetic confirmation.
  exponent collapses into the band       => te13's flat read was a calibration artefact;
                                            the instrument is validated on constructed
                                            perturbations ONLY, and the write-up must say
                                            so. Rule #3' still stands on C1/C2/C3.

`ji` is re-read on the same honest instrument in the same run, so the two are directly
comparable and the sign discrepancy is attributed rather than left hanging.

Run: python src/probe_mslope_te13_honest.py -> results/probe_mslope_te13_honest.txt
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from fast_dw import ba_fast, fit_dw
from train_common import RESULTS_DIR, load_dataset

N_SPLITS = 6
# ⚠️ SUPERSEDED AS A VERDICT THRESHOLD by `probe_mslope_c1check.py` (Phase 1b): the C2
# selection-noise band is STRONGLY INSTRUMENT-DEPENDENT (-0.33 w_stock / -0.85 w_chal /
# -1.54 argmax), so these numbers are specific to the w_stock instrument and are NOT a
# universal null. The robust anchors are C1 ≈ 0 (information) and C3 ≈ -1 (variance
# reduction), which separate by 3.4 sd even under calibration-free argmax.
# READ VERDICTS BY DISTANCE FROM C1 AND C3, not from this band; and say "not information"
# rather than "variance reduction" when a negative overlaps both. The VERDICT strings
# printed below are kept for continuity with the Day-22 logs — no verdict changes under
# the corrected reading.
NULL_LO, NULL_HI = -0.773, -0.060
OUT = RESULTS_DIR / "probe_mslope_te13_honest.txt"

ARMS: list[tuple[str, list[str], list[str]]] = [
    (
        "te13 recipe (KNOWN REAL)",
        [f"realmlp_te13_u_s{s}" for s in (42, 7, 123)],
        [f"realmlp_u_s{s}" for s in (42, 7, 123)],
    ),
    (
        "ji (joint basin x IMP)",
        [f"realmlp_te13_imp_u_s{s}ji" for s in (42, 7, 123, 2024)],
        [f"realmlp_te13_u_s{s}" for s in (42, 7, 123, 2024)],
    ),
]


def _decide(y_fit, p_fit, p_eval, y_eval) -> float:  # noqa: ANN001
    """Honest per-arm decision layer: fit w on the fit half, score on the eval half.

    Uses `fast_dw.fit_dw` (coarse-to-fine grid) rather than `tune_decision_weights`
    (Nelder-Mead): on this piecewise-constant objective the grid is ~30x faster AND finds
    a strictly better optimum (+0.000077 BA on the deployed 8-seed breadth, verified in
    `fast_dw.selftest`). Identical fitter on both arms, so it is common-mode in the delta.
    """
    dw = fit_dw(y_fit, p_fit)
    counts = np.bincount(y_eval, minlength=p_eval.shape[1])
    return ba_fast(y_eval, np.argmax(p_eval * dw, axis=1), counts)


def _cell(C, S, y, f, e, m_max):  # noqa: ANN001, ANN202
    """One (fit, eval) cell: honest per-arm dw delta at every M, averaged over ALL subsets
    of that size (exact, and far cheaper than enumerating orderings — every ordering
    prefix of length m is a subset of size m, and each subset is scored once)."""
    out = np.zeros(m_max)
    for m in range(1, m_max + 1):
        subs = list(itertools.combinations(range(m_max), m))
        for idx in subs:
            cm, sm = C[list(idx)].mean(0), S[list(idx)].mean(0)
            out[m - 1] += _decide(y[f], cm[f], cm[e], y[e]) - _decide(y[f], sm[f], sm[e], y[e])
        out[m - 1] /= len(subs)
    return out


def _exponent(ms, d) -> float:  # noqa: ANN001
    if np.any(np.asarray(d) <= 0):
        return float("nan")
    return float(np.polyfit(np.log(ms), np.log(d), 1)[0])


def main() -> None:
    ds = load_dataset()
    y = ds.y
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    cells = []
    for seed in range(N_SPLITS):
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                           random_state=seed).split(y, y))
        cells += [(a, b), (b, a)]

    say("=" * 78)
    say("probe_mslope_te13_honest -- per-arm HONEST dw (fit half -> eval half), 12 cells")
    say(f"C2 null band from probe_mslope.py: [{NULL_LO:+.3f}, {NULL_HI:+.3f}]")
    say("=" * 78)

    for label, ck, sk in ARMS:
        C = np.stack([np.load(RESULTS_DIR / f"oof_{k}.npy") for k in ck])
        S = np.stack([np.load(RESULTS_DIR / f"oof_{k}.npy") for k in sk])
        m_max = len(C)
        res = Parallel(n_jobs=6)(delayed(_cell)(C, S, y, f, e, m_max) for f, e in cells)
        A = np.asarray(res)
        mean = A.mean(0)
        ms = np.arange(1, m_max + 1, dtype=float)
        exp = _exponent(ms, mean)
        boots = []
        rng = np.random.default_rng(20260729)
        for _ in range(400):
            i = rng.integers(0, len(A), len(A))
            boots.append(_exponent(ms, A[i].mean(0)))
        b = np.asarray(boots)
        b = b[np.isfinite(b)]
        say(f"-- {label}  ({m_max} seeds, {len(cells)} cells, all subsets per size)")
        say("   M       " + "".join(f"    {int(m)}     " for m in ms))
        say("   delta   " + "".join(f"  {v:+.6f}" for v in mean))
        say("   sd      " + "".join(f"  {v:+.6f}" for v in A.std(0) / np.sqrt(len(A))))
        if mean[0] != 0:
            say("   norm.   " + "".join(f"  {v / mean[0]:+.6f}" for v in mean))
        say(f"   EXPONENT {exp:+.3f}  (bootstrap sd {b.std() if len(b) else float('nan'):.3f}, "
            f"{len(b)}/400 finite)")
        if np.isnan(exp):
            v = "UNDEFINED (delta crosses zero)"
        elif exp > NULL_HI:
            v = "MEAN SHIFT -- does not dilute"
        elif exp < NULL_LO:
            v = "VARIANCE REDUCTION"
        else:
            v = "INSIDE THE NULL BAND"
        say(f"   VERDICT: {v}")
        say()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
