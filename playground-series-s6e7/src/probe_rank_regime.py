"""Day-13 PM — rank-regime decision table (advisor gap: score-MC != rank objective).

Every prior finals instrument maximizes E[max(S1,S2)] with no opponent. Final
standing is P(max(S1,S2) > best rival's private score). This parameterizes the
rival instead of reading the LB (protocol-clean): rival private score
R = mu_u4 + d_lead + eps, eps ~ N(0, sigma_priv), optionally correlated (rho)
with our own private draw through the shared label-noise realization.

Output: P(pair beats rival) for each candidate #2, over d_lead x rho grids —
a lookup table the user applies with their own knowledge of the public band
(discounted by the measured lbprobe contamination ruler, results/
lbprobe_ruler.txt: >= +0.00076 of public is fakeable). Regime logic expected:
  - clear lead (d_lead << 0): all pairs ~tie; pick #2 by mean (clone fine)
  - knife edge: max-P(win) pair = highest-variance-with-mean (realmlp breadth)
  - trailing: variance dominates; most-decorrelated #2 (champion) pays

Run: python src/probe_rank_regime.py -> results/probe_rank_regime.txt
"""

from __future__ import annotations

import sys
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

A_KEY = "lrstack_u4_r"  # final #1
SECONDS = [
    "realmlp_te13_u_breadth",
    "lrstack6b_r",
    "champion_v1",
    "lrstack_u3_r",
]
SIGMA_PRIV = 0.0006
D_LEAD = [-0.0010, -0.0005, -0.0002, 0.0, 0.0002, 0.0005, 0.0010]
RHOS = [0.0, 0.5, 0.9]
SEED = 20260715


def main() -> None:
    ds = load_dataset()
    y = ds.y
    lab_a = labels_of(corrected(A_KEY, "oof"))
    rng = np.random.default_rng(SEED)

    lines = [
        f"rank-regime table: P(max({A_KEY}, #2) > rival) ; rival = mu_a + d_lead"
        f" + N(0,{SIGMA_PRIV}) with corr rho to our private draw",
        "d_lead > 0 means the rival's true mean is AHEAD of ours.",
        "(apply the lbprobe ruler: public leads <= ~0.0008 are fakeable;"
        " see results/lbprobe_ruler.txt)",
        "",
    ]
    for b_key in SECONDS:
        lab_b = (
            champion_oof_labels(y)
            if b_key == "champion_v1"
            else labels_of(corrected(b_key, "oof"))
        )
        a_priv, b_priv, _, _ = mc_cells(y, lab_a, lab_b)
        mu_a = float(a_priv.mean())
        pair_max = np.maximum(a_priv, b_priv)
        # standardized own-draw for the correlated rival component
        z_own = (pair_max - pair_max.mean()) / max(pair_max.std(), 1e-12)
        lines.append(f"#2 = {b_key}  (E[S2]={b_priv.mean():.5f}, "
                     f"E[max]={pair_max.mean():.5f})")
        header = "  d_lead:" + "".join(f"{d:+9.4f}" for d in D_LEAD)
        lines.append(header)
        for rho in RHOS:
            eps = rho * z_own * SIGMA_PRIV + np.sqrt(1 - rho**2) * rng.normal(
                0, SIGMA_PRIV, len(pair_max)
            )
            row = []
            for d in D_LEAD:
                rival = mu_a + d + eps
                row.append(float((pair_max > rival).mean()))
            lines.append(
                f"  rho={rho:.1f} " + "".join(f"{p:9.3f}" for p in row)
            )
        lines.append("")
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_rank_regime.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
