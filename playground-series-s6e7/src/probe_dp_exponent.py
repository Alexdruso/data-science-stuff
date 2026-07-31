"""Day-23: the M-slope of `dp` — driver POSTERIORS, the strongest reconstructable driver.

Day-22 priced `imp` (sleep reconstruction, scalar point estimate) at exponent **-0.850**
= "not information", and re-characterised Day-17's "REAL channel" as a variance reducer.
But `imp` reconstructs the WEAKEST of the three drivers. `train_sm.py`'s free diagnostic
found `physical_activity_level` is by far the most recoverable (logloss gain +0.5515,
top-1 0.712 vs 0.339 prior) and it had never been fed to a strong leg as features.

THE QUESTION: does driver STRENGTH change the exponent, or is the whole reconstruction
family a variance reducer regardless of how much driver signal is recoverable? Either
answer is informative, which is the bar for spending the GPU.

Arm `dp` = realmlp_te13 + dp_features.dp_block (20 cols: P(driver|x_obs) for all three
drivers + the three rule-band masses of the sleep posterior). Seeds 42/7/123 — matched
with the stock arm AND with `imp`'s 3 seeds, so all three exponents are comparable.

Instrument: the SAME honest per-arm decision layer as `probe_mslope_te13_honest.py`
(weights fitted on the fit half, scored on the eval half, 12 cells, all subsets per
size). `dp` vs stock is a two-matrix comparison, so frozen-w would inherit the
calibration confound documented in Phase 2b.

⚠️ PRE-REGISTERED BARS (comp CLAUDE.md Day-23 block; NOT to be moved after the read):
  exponent <= -0.060 (C2 null band's upper edge, i.e. at/below `imp` -0.850 or C3 -1.002)
      => DEAD regardless of its M=1 size. No escalation, no level read.
  exponent > 0 (the te13 +0.558 / C1 information end)
      => escalate: extend to the deployed M and read the level.
  A #2 swap additionally needs >= +0.001 at the deployed M + a submission slot +
  finals_audit pass + byte verification. A TIE MEANS NO SWAP.

⚠️ HONEST PRIOR, stated before the number lands: expect a clean negative. (1) `imp` read
-0.850 and this is the same family. (2) Day-7's `train_dp.py` already measured this exact
channel at -0.0002 solo (pre-te13, repaired surface, LGBM consumer). (3) The motivating
+0.5515/0.712 diagnostic is IN-SAMPLE and fitted on train+test POOLED, and activity's mask
is trigger-coupled in train but uniform in test — the Day-4 shift sits on this channel.
Reporting a clean negative against unmoved bars IS the successful outcome.

Run: python src/probe_dp_exponent.py -> results/probe_dp_exponent.txt
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
OUT = RESULTS_DIR / "probe_dp_exponent.txt"

# Day-22 anchors, for stating the verdict by distance from the two ROBUST ends of the
# scale rather than from the instrument-specific C2 band (Phase 1b correction).
ANCHORS = {"te13 (information)": +0.558, "C1 oracle mean shift": -0.027,
           "imp (sleep recon.)": -0.850, "C3 variance reduction": -1.002,
           "ji (basin x imp)": -1.107}


def main() -> None:
    y = load_dataset().y
    C = np.stack([np.load(RESULTS_DIR / f"oof_realmlp_te13_dp_u_s{s}.npy") for s in SEEDS])
    S = np.stack([np.load(RESULTS_DIR / f"oof_realmlp_te13_u_s{s}.npy") for s in SEEDS])
    assert C.shape == S.shape, f"arm shape mismatch {C.shape} vs {S.shape}"

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
    rng = np.random.default_rng(20260731)
    b = np.asarray([_exponent(ms, res[rng.integers(0, len(res), len(res))].mean(0))
                    for _ in range(400)])
    b = b[np.isfinite(b)]
    sd = float(b.std()) if len(b) else float("nan")

    lines = [
        "=" * 78,
        "probe_dp_exponent -- `dp` (label-free driver POSTERIORS, 20 cols), honest per-arm dw",
        f"seeds {SEEDS}, {len(cells)} cells, all subsets per size",
        f"C2 null band (w_stock-specific): [{NULL_LO:+.3f}, {NULL_HI:+.3f}]",
        "robust anchors: " + "  ".join(f"{k} {v:+.3f}" for k, v in ANCHORS.items()),
        "=" * 78,
        "   M       " + "".join(f"    {int(m)}     " for m in ms),
        "   delta   " + "".join(f"  {v:+.6f}" for v in mean),
        "   se      " + "".join(f"  {v:+.6f}" for v in res.std(0) / np.sqrt(len(res))),
    ]
    if mean[0] != 0:
        lines.append("   norm.   " + "".join(f"  {v / mean[0]:+.6f}" for v in mean))
    lines.append(f"   EXPONENT {exp:+.3f}  (bootstrap sd {sd:.3f}, {len(b)}/400 finite)")
    if np.isfinite(exp) and np.isfinite(sd) and sd > 0:
        lines.append("   distance from anchors (sd): " + "  ".join(
            f"{k.split(' ')[0]} {(exp - v) / sd:+.1f}" for k, v in ANCHORS.items()))

    # PRE-REGISTERED verdict rule -- written before the number, not chosen after it.
    if np.isnan(exp):
        v = ("UNDEFINED (delta crosses zero) => DEAD by the pre-registered bar "
             "(an exponent that cannot be fit is not evidence of information)")
    elif exp > 0:
        v = ("ESCALATE -- exponent above zero, the information end of the scale. "
             "Extend to the deployed M and read the LEVEL against the +0.001 bar.")
    elif exp <= NULL_HI:
        v = ("DEAD -- at or below the null band's upper edge. Driver STRENGTH does not "
             "rescue the reconstruction family; it behaves like `imp`/C3 regardless. "
             "No escalation, no level read, per the pre-registered bar.")
    else:
        v = ("INCONCLUSIVE -- between the null-band edge and zero. No escalation "
             "(the bar required exponent > 0). Report, do not adopt.")
    lines.append(f"   VERDICT: {v}")

    report = "\n".join(lines)
    print(report, flush=True)
    OUT.write_text(report + "\n")


if __name__ == "__main__":
    main()
