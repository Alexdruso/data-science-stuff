"""Finals-pair selection: E[max(pair)] over all candidate pairs (Day-9 pm).

User-directed public reads (scores unrecorded) showed the whole repaired
lineage clearing champion_v1 and realmlp_rule_r_s42 stack-competitive as a
solo. This runs the honest instrument over every candidate pair: the same
cell-MC as probe_emax_portfolio (multinomial test emulation + mv-hypergeometric
private split over (class x corr_a x corr_b) cells), reporting E[max(pair)] on
private, P(a beats b), and each candidate's solo E[score].

Caveat printed with the result: repaired-surface OOF understates test accuracy
relative to the unrepaired champion (repair emulates the test mechanism ==
harder inputs; champion OOF has the easy train mechanism), so cross-surface
comparisons are conservative AGAINST the repaired lineage — the public sweep
suggests the truth is more favorable to it than these numbers.

Run: python src/probe_finals_pairs.py  -> results/probe_finals_pairs.txt
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_emax_portfolio import (
    RESULTS_DIR,
    champion_oof_labels,
    corrected,
    labels_of,
    mc_cells,
)
from train_common import load_dataset

CANDIDATES = [
    "champion_v1",
    "lrstack6_r",
    "metablend_r",
    "lrstack_u_r",
    "hgbc_te3_u_breadth",
    "metablend_u",
]


def main() -> None:
    ds = load_dataset()
    y = ds.y
    labs = {}
    for c in CANDIDATES:
        if c == "champion_v1":
            labs[c] = champion_oof_labels(y)
        else:
            labs[c] = labels_of(corrected(c, "oof"))

    out: list[str] = []
    out.append("finals-pair E[max] on emulated private (B=20k); OOF-label instrument")
    out.append("NOTE: repaired-lineage candidates are UNDERRATED here (harder OOF surface")
    out.append("than champion's); public sweep favors them beyond these numbers.\n")

    solo: dict[str, float] = {}
    rows = []
    for a, b in combinations(CANDIDATES, 2):
        a_p, b_p, _, _ = mc_cells(y, labs[a], labs[b])
        emax = float(np.maximum(a_p, b_p).mean())
        solo[a] = float(a_p.mean())
        solo[b] = float(b_p.mean())
        rows.append((emax, a, b, float((a_p > b_p).mean())))

    out.append(f"{'candidate':<22} {'E[priv]':>9}")
    for c in CANDIDATES:
        out.append(f"  {c:<20} {solo[c]:>9.5f}")
    out.append("")
    out.append(f"{'pair':<44} {'E[max]':>9} {'P(first wins)':>14}")
    for emax, a, b, pa in sorted(rows, reverse=True):
        out.append(f"  {a + ' + ' + b:<42} {emax:>9.5f} {pa:>14.3f}")

    best = max(rows)
    out.append(
        f"\nBEST PAIR by honest E[max]: {best[1]} + {best[2]}  E[max]={best[0]:.5f}"
    )
    report = "\n".join(out)
    print(report)
    (RESULTS_DIR / "probe_finals_pairs.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
