"""Menu-level label fusion (Day-18 evening pitch P2) — zero-retrain.

Every measured combination fuses PROBAS upstream (blend weights, LR meta,
breadth means). A label-level majority vote across the deployed finals-menu
candidates is a deployment-altitude cell never measured. Pre-registered:
vote over the 6 menu candidates' OOF labels, ties broken toward lrstack_u4_r
(the strongest member); read = OOF bacc + mc_cells E[priv] vs u4. Menu entry
iff E[priv] beats u4's beyond MC noise (~1e-4) — else the last combination
altitude is measured-closed.

Run: python src/probe_menu_vote.py -> results/probe_menu_vote.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from probe_emax_portfolio import champion_oof_labels, corrected, labels_of, mc_cells
from train_common import RESULTS_DIR, load_dataset

MENU = [
    "lrstack_u4_r",
    "lrstack6b_r",
    "realmlp_te13_u_breadth",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
]
ANCHOR = "lrstack_u4_r"


def main() -> None:
    ds = load_dataset()
    y = ds.y
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    labs = {c: labels_of(corrected(c, "oof")) for c in MENU}
    labs["champion_v1"] = champion_oof_labels(y)
    names = [*MENU, "champion_v1"]
    stack = np.stack([labs[c] for c in names])
    # majority vote, ties -> anchor's label
    counts = np.stack([(stack == k).sum(axis=0) for k in range(3)])
    top = counts.max(axis=0)
    vote = counts.argmax(axis=0)
    tied = (counts == top).sum(axis=0) > 1
    vote[tied] = labs[ANCHOR][tied]
    say(f"voters: {names}; ties -> {ANCHOR}: {int(tied.sum())} rows")
    for c in names:
        say(f"  {c:<24} OOF bacc {balanced_accuracy_score(y, labs[c]):.5f}  "
            f"(disagrees with vote on {int((labs[c] != vote).sum())} rows)")
    say(f"  {'VOTE':<24} OOF bacc {balanced_accuracy_score(y, vote):.5f}")

    a_p, b_p, _, _ = mc_cells(y, vote, labs[ANCHOR])
    say(f"\nmc_cells: E[priv] vote {float(a_p.mean()):.5f} vs {ANCHOR} "
        f"{float(b_p.mean()):.5f}  P(vote wins) {float((a_p > b_p).mean()):.3f}")
    say("menu entry iff vote beats u4 E[priv] beyond ~1e-4; else altitude closed")
    (RESULTS_DIR / "probe_menu_vote.txt").write_text("\n".join(lines) + "\n")
    print(f"report -> {RESULTS_DIR / 'probe_menu_vote.txt'}", flush=True)


if __name__ == "__main__":
    main()
