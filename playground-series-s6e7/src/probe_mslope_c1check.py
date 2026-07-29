"""Day-22: is C1's flat exponent robust to the DECISION-LAYER CALIBRATION choice?

`probe_mslope.py` read every control under `w` frozen on the STOCK arm. For C2 (two seed
halves of the same leg) and C3 (shrink toward a disjoint member average, identical mean)
that is neutral — both arms share a calibration. **For C1 it is not**: mixing toward
`onehot(y)` adds probability mass in proportion to each class's frequency, and at-risk is
85.9% of rows, so C1's optimal `w` shifts — exactly as `ji`'s did (stock `[1, 0.874,
0.900]` vs ji `[1, 1.161, 1.101]` renormalised).

C1 is the ONLY control anchoring the "exponent ≈ 0" end of the scale, so that caveat has
to be measured, not argued away. Three instruments per control:

  w_stock   `w` frozen on the stock breadth      (probe_mslope.py's original)
  w_chal    `w` frozen on the CHALLENGER breadth (the mirror image)
  argmax    no decision layer at all             (CALIBRATION-FREE — the decisive one)

If C1 reads ≈0 under all three, the M-independence of a fixed calibration penalty is
measured and C1 is bulletproof. If it only reads ≈0 under one, the scale is an artefact.

C2 and C3 are re-read on the same three instruments as a consistency check.

Run: python src/probe_mslope_c1check.py -> results/probe_mslope_c1check.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from fast_dw import fit_dw
from train_common import RESULTS_DIR, load_dataset

SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777]
LEG = "realmlp_te13_u"
M_MAX = 4
N_REPS = 40
RNG_SEED = 20260729
LAM1 = 0.012278  # calibrated in probe_mslope.py to read +0.0003 at M=1
LAM3 = 0.499999
OUT = RESULTS_DIR / "probe_mslope_c1check.txt"


def _ba(y, p, w) -> float:  # noqa: ANN001
    return float(balanced_accuracy_score(y, np.argmax(p if w is None else p * w, axis=1)))


def _exponent(ms, d) -> float:  # noqa: ANN001
    d = np.asarray(d)
    if np.any(d <= 0):
        return float("nan")
    return float(np.polyfit(np.log(ms), np.log(d), 1)[0])


def main() -> None:
    ds = load_dataset()
    y = ds.y
    P = np.stack([np.load(RESULTS_DIR / f"oof_{LEG}_s{s}.npy").astype(np.float32) for s in SEEDS])
    n = P.shape[1]
    onehot = np.zeros((n, 3), dtype=np.float32)
    onehot[np.arange(n), y] = 1.0

    w_stock = np.asarray(json.loads((RESULTS_DIR / f"mslope_w_{LEG}.json").read_text()))
    c1_full = (1 - LAM1) * P.mean(0) + LAM1 * onehot
    cache = RESULTS_DIR / "mslope_w_c1chal.json"
    if cache.exists():
        w_c1 = np.asarray(json.loads(cache.read_text()))
    else:
        w_c1 = fit_dw(y, c1_full.astype(np.float64))
        cache.write_text(json.dumps(w_c1.tolist()))

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say("=" * 78)
    say("probe_mslope_c1check -- is the control scale robust to the decision-layer choice?")
    say("=" * 78)
    say(f"w frozen on stock breadth-8      : {np.round(w_stock, 5).tolist()}")
    say(f"w frozen on the C1-perturbed arm : {np.round(w_c1, 5).tolist()}")
    say(f"   => C1 DOES shift the optimal decision weights, as suspected "
        f"(max|dw| = {np.abs(w_c1 - w_stock).max():.4f})")
    say()

    rng = np.random.default_rng(RNG_SEED)
    ms = np.arange(1, M_MAX + 1, dtype=float)
    instruments = {"w_stock": w_stock, "w_chal(C1)": w_c1, "argmax": None}
    traj: dict[tuple[str, str], list[list[float]]] = {
        (c, i): [] for c in ("C1", "C2", "C3") for i in instruments
    }

    for _ in range(N_REPS):
        perm = rng.permutation(8)
        arm, tgt = perm[:4], perm[4:]
        tgt_avg = P[tgt].mean(axis=0)
        rows: dict[tuple[str, str], list[float]] = {k: [] for k in traj}
        # C2's challenger half is picked by the M=1 read, per instrument
        for iname, w in instruments.items():
            hi, lo = (arm, tgt) if _ba(y, P[arm[0]], w) >= _ba(y, P[tgt[0]], w) else (tgt, arm)
            for m in range(1, M_MAX + 1):
                sub = P[arm[:m]]
                stock = sub.mean(axis=0)
                s_ba = _ba(y, stock, w)
                rows[("C1", iname)].append(
                    _ba(y, (1 - LAM1) * stock + LAM1 * onehot, w) - s_ba
                )
                rows[("C3", iname)].append(
                    _ba(y, ((1 - LAM3) * sub + LAM3 * tgt_avg[None]).mean(axis=0), w) - s_ba
                )
                rows[("C2", iname)].append(
                    _ba(y, P[hi[:m]].mean(0), w) - _ba(y, P[lo[:m]].mean(0), w)
                )
        for k in traj:
            traj[k].append(rows[k])

    say(f"-- exponents over M=1..{M_MAX}, {N_REPS} replicates, bootstrap sd ------------")
    say(f"   {'control':<6} {'instrument':<12} {'M=1':>10} {'M=4':>10} {'exponent':>10} {'sd':>7}")
    for c in ("C1", "C2", "C3"):
        for iname in instruments:
            A = np.asarray(traj[(c, iname)])
            mean = A.mean(axis=0)
            e = _exponent(ms, mean)
            b = []
            for _ in range(400):
                idx = rng.integers(0, len(A), len(A))
                b.append(_exponent(ms, A[idx].mean(axis=0)))
            b = np.asarray(b)
            b = b[np.isfinite(b)]
            say(f"   {c:<6} {iname:<12} {mean[0]:>+10.6f} {mean[-1]:>+10.6f} "
                f"{e:>+10.3f} {(b.std() if len(b) else float('nan')):>7.3f}")
        say()

    say("READ: if C1's exponent is ~0 under ALL THREE instruments -- in particular under")
    say("      plain argmax, which has no decision layer to be confounded by -- then the")
    say("      'mean shifts do not dilute' end of the scale is measured, not asserted.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
