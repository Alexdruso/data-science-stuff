"""Day-21 pre-registration #2: does the ji recipe delta survive AVERAGING?

Final #2 (`realmlp_te13_u_breadth`) is a SOLO leg — no combiner — so veto #18's
absorption mechanism does not apply to it and the +0.000198 single-seed recipe
delta passes through to a solo submission by construction.  The open question is
whether it survives breadth averaging, which every escalated delta in this
campaign has failed to do at full size (6/6 shrank ~50%).

Matched-N canary: stock-breadth-4 is free (all stock per-seed OOFs are on disk);
ji-breadth-4 costs 3 GPU runs instead of 7.

PRE-REGISTERED BARS on the paired dw-OOF delta (ji_b4 - stock_b4), see CLAUDE.md:
  >= +0.00010 and sign-consistent   extend to N=8 for a deliverable
  <= +0.00002                        does not survive averaging; stop
  otherwise                          inconclusive, no swap

Reported alongside: the same delta at each prefix N=1..4, so the shrinkage
TRAJECTORY is visible rather than just its endpoint, and a per-seed table so a
single lucky seed can't carry the mean unnoticed.

Run: python src/probe_ji_canary.py -> results/probe_ji_canary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict
from sklearn.metrics import balanced_accuracy_score

SEEDS = [42, 7, 123, 2024]
STOCK = "realmlp_te13_u_s{seed}"
JI = "realmlp_te13_imp_u_s{seed}ji"


def _oof(key: str) -> np.ndarray | None:
    p = RESULTS_DIR / f"oof_{key}.npy"
    return np.load(p) if p.exists() else None


def _dw_bacc(y: np.ndarray, probs: np.ndarray) -> float:
    dw = np.asarray(tune_decision_weights(y, probs))
    return float(balanced_accuracy_score(y, weighted_predict(probs, dw)))


def main() -> None:
    y = load_dataset().y

    stock = {s: _oof(STOCK.format(seed=s)) for s in SEEDS}
    ji = {s: _oof(JI.format(seed=s)) for s in SEEDS}
    missing = [s for s in SEEDS if stock[s] is None or ji[s] is None]
    if missing:
        raise SystemExit(f"missing arrays for seeds {missing} — canary not ready")

    lines = [
        "ji CANARY (Day-21 pre-reg #2) — does the recipe delta survive averaging?",
        f"seeds {SEEDS}; solo leg, no combiner (veto #18 does not apply)",
        "PRE-REG BARS on ji_b4 - stock_b4: >=+0.00010 extend / <=+0.00002 stop / "
        "else inconclusive",
        "",
        "per-seed (dw-weighted OOF balanced acc):",
    ]
    per_seed = []
    for s in SEEDS:
        a, b = _dw_bacc(y, stock[s]), _dw_bacc(y, ji[s])
        per_seed.append(b - a)
        lines.append(f"  s{s:<5} stock {a:.6f}   ji {b:.6f}   delta {b - a:+.6f}")
    ps = np.array(per_seed)
    lines.append(
        f"  per-seed mean {ps.mean():+.6f}  sd {ps.std(ddof=1):.6f}  "
        f"se {ps.std(ddof=1) / np.sqrt(len(ps)):.6f}  positive {int((ps > 0).sum())}/{len(ps)}"
    )

    lines += ["", "breadth prefixes (the shrinkage trajectory):"]
    traj = []
    for n in range(1, len(SEEDS) + 1):
        use = SEEDS[:n]
        a = _dw_bacc(y, np.mean([stock[s] for s in use], axis=0))
        b = _dw_bacc(y, np.mean([ji[s] for s in use], axis=0))
        traj.append(b - a)
        lines.append(
            f"  N={n}  stock {a:.6f}   ji {b:.6f}   delta {b - a:+.6f}   seeds {use}"
        )

    d4 = traj[-1]
    lines += ["", f"HEADLINE  ji_b4 - stock_b4 = {d4:+.6f}"]
    if d4 >= 0.00010 and ps.mean() > 0:
        lines.append("VERDICT: EXTEND to N=8 — delta survives averaging at the bar.")
    elif d4 <= 0.00002:
        lines.append("VERDICT: STOP — delta does not survive averaging. #2 unchanged.")
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no swap. #2 unchanged.")
    lines.append(
        "NOTE: #1 stays lrstack_u4_r regardless (OOF cannot rank <0.001). Any #2 "
        "successor still needs slot + finals_audit + byte-verify to be clickable."
    )

    out = "\n".join(lines)
    print(out)
    (RESULTS_DIR / "probe_ji_canary.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
