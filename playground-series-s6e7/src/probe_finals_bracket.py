"""δ-bracket finals MC (Day-10 PM, decision-theorist audit item #1).

The planned pair lrstack_u3 + lrstack6b_r has hedge value +0.00000 at face on
the mc_cells instrument, and only pays if the repaired candidate's OOF
understates its true test mean by δ_r ≥ ~0.0012 — a knife-edge INSIDE the
untested "reads 0.001-0.002 low by construction" band. This probe brackets
the decision instead of assuming it:

- b = lrstack6b_r (repaired surface): its OOF val rows carry REPAIR-ADDED
  NaNs the real test rows don't have. Bracket its mean shift by flipping a
  fraction f of its WRONG rows on the repair-added-NaN set to correct
  (δ_r = induced OOF bacc gain; f swept 0..0.4 ~ up to the clean-input
  ceiling). The added-NaN set is exactly reproducible via
  train_common._uniform_remask's fixed RNG.
- a = lrstack_u3 (unrepaired surface): its OOF miss-driver rows have TRIGGER
  placement while test is uniform — Day-4's placement term bounds the
  overstatement at ~−0.001. Bracket by flipping a fraction g of its CORRECT
  rows on the 4-col-NaN set to wrong (δ_u swept 0..−0.001).

For each (f, g) grid cell: rerun the joint cell-MC → E[max(pair)], P(a wins),
and the hedge bonus vs solo-a. Output = the (δ_r, δ_u) region where keeping
lrstack6b_r as final #2 beats the all-u pair.

Run: python src/probe_finals_bracket.py -> results/probe_finals_bracket.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_emax_portfolio import RESULTS_DIR, corrected, labels_of, mc_cells
from sklearn.metrics import balanced_accuracy_score
from train_common import MECHANISM_SHIFTED, _uniform_remask, load_dataset

A_KEY = "lrstack_u3_r"  # unrepaired stack (file suffix _r = raw-variant tag)
B_KEY = "lrstack6b_r"  # repaired stack
ALT_KEY = "lrstack_u2_r"  # the all-u alternative #2
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
G_GRID = [0.0, 0.05, 0.1]
SEED = 7


def synth_labels(y: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Labels with the given correctness pattern (wrong class choice is
    irrelevant to mc_cells, which only reads correctness)."""
    lab = np.where(corr, y, (y + 1) % 3)
    return lab.astype(np.int64)


def main() -> None:
    ds = load_dataset()
    y = ds.y
    rng = np.random.default_rng(SEED)

    cols = [c for c in MECHANISM_SHIFTED if c in ds.train.columns]
    raw_nan = ds.train[cols].isna()
    repaired = _uniform_remask(ds.train.copy(), ds.test)
    added = (repaired[cols].isna() & ~raw_nan).any(axis=1).to_numpy()
    anynan = raw_nan.any(axis=1).to_numpy()
    print(f"repair-added-NaN rows {added.sum()} ({added.mean():.1%}); "
          f"raw 4-col-NaN rows {anynan.sum()} ({anynan.mean():.1%})")

    corr_a0 = labels_of(corrected(A_KEY, "oof")) == y
    corr_b0 = labels_of(corrected(B_KEY, "oof")) == y
    corr_alt = labels_of(corrected(ALT_KEY, "oof")) == y
    bacc = lambda c: balanced_accuracy_score(y, synth_labels(y, c))  # noqa: E731
    base_a, base_b = bacc(corr_a0), bacc(corr_b0)

    # the all-u alternative pair (a + alt), fixed reference
    a_p, alt_p, _, _ = mc_cells(y, synth_labels(y, corr_a0), synth_labels(y, corr_alt))
    emax_alt = float(np.maximum(a_p, alt_p).mean())

    lines = [
        f"δ-bracket finals MC: a={A_KEY} (OOF {base_a:.5f}), b={B_KEY} (OOF {base_b:.5f})",
        f"reference all-u pair a+{ALT_KEY}: E[max]={emax_alt:.5f}",
        "",
        f"{'δ_u':>8} {'δ_r':>8} {'E[a]':>9} {'E[b]':>9} {'E[max]':>9} "
        f"{'P(a>b)':>7} {'vs all-u pair':>13}",
    ]
    for g in G_GRID:
        corr_a = corr_a0.copy()
        cand = np.where(anynan & corr_a0)[0]
        kill = rng.choice(cand, size=int(g * len(cand)), replace=False)
        corr_a[kill] = False
        d_u = bacc(corr_a) - base_a
        for f in F_GRID:
            corr_b = corr_b0.copy()
            cand_b = np.where(added & ~corr_b0)[0]
            fix = rng.choice(cand_b, size=int(f * len(cand_b)), replace=False)
            corr_b[fix] = True
            d_r = bacc(corr_b) - base_b
            ap, bp, _, _ = mc_cells(
                y, synth_labels(y, corr_a), synth_labels(y, corr_b)
            )
            emax = float(np.maximum(ap, bp).mean())
            lines.append(
                f"{d_u:>+8.4f} {d_r:>+8.4f} {ap.mean():>9.5f} {bp.mean():>9.5f} "
                f"{emax:>9.5f} {float((ap > bp).mean()):>7.3f} "
                f"{emax - emax_alt:>+13.5f}"
            )
        lines.append("")

    lines.append(
        "read: 'vs all-u pair' > 0 means keeping the repaired hedge beats the "
        f"a+{ALT_KEY} pair at that (δ_u, δ_r)."
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_finals_bracket.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
