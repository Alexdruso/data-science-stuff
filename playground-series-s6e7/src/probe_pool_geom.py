"""Member-pooling geometry probe (Day-18 lens A, A1) — zero-retrain.

The combiner matrix (Day-13) swept the META over FIXED leg arrays; every leg
array is itself an ARITHMETIC mean of member softmaxes (breadth seeds). The
pooling FUNCTION is an unswept axis: logit-mean (geometric) reweights member
confidence, median/trimmed kill member outliers. Pure array math over the
on-disk per-seed OOF arrays.

Arms per family (realmlp_te13_u ×12 seeds, hgbc_te3_u ×8 seeds):
  arith (incumbent, reproduced from members) / logit / median / trimmed
Instruments: OOF argmax bacc (weight-free) + robust-weight refit bacc per arm
(the house deterministic instrument). Screen bar +0.0007 (1-seed-canary
convention applies per family); escalation = repooled leg arrays -> stack
rebuild + 12-cell seat.

Run: python src/probe_pool_geom.py -> results/probe_pool_geom.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

from data_science_stuff.kaggle_utils import weighted_predict

FAMILIES: dict[str, list[str]] = {
    "realmlp_te13_u": [
        "s42", "s7", "s123", "s99", "s31", "s2024", "s2025", "s7777",
        "s101", "s202", "s303", "s404",
    ],
    "hgbc_te3_u": ["s42", "s7", "s123", "s99", "s31", "s2024", "s2025", "s7777"],
}
EPS = 1e-7


def pool(members: np.ndarray, how: str) -> np.ndarray:
    if how == "arith":
        out = members.mean(axis=0)
    elif how == "logit":
        out = np.exp(np.log(np.clip(members, EPS, 1.0)).mean(axis=0))
    elif how == "median":
        out = np.median(members, axis=0)
    elif how == "trimmed":
        srt = np.sort(members, axis=0)
        out = srt[1:-1].mean(axis=0)
    else:
        raise ValueError(how)
    return out / out.sum(axis=1, keepdims=True)


def main() -> None:
    ds = load_dataset()
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    for fam, seeds in FAMILIES.items():
        members = np.stack(
            [np.load(RESULTS_DIR / f"oof_{fam}_{s}.npy") for s in seeds]
        )
        say(f"\n== {fam} ({len(seeds)} members) ==")
        base_arg = base_wt = None
        for how in ("arith", "logit", "median", "trimmed"):
            p = pool(members, how)
            w = robust_decision_weights(ds.y, p)
            arg = balanced_accuracy_score(ds.y, p.argmax(axis=1))
            wt = balanced_accuracy_score(ds.y, weighted_predict(p, w))
            if how == "arith":
                base_arg, base_wt = arg, wt
                say(f"  {how:8s} argmax {arg:.5f}  wt {wt:.5f}  (incumbent)")
            else:
                say(
                    f"  {how:8s} argmax {arg:.5f} ({arg - base_arg:+.5f})  "
                    f"wt {wt:.5f} ({wt - base_wt:+.5f})"
                )
    say("\nscreen bar +0.0007; escalation = repooled legs -> stack + 12-cell seat")
    (RESULTS_DIR / "probe_pool_geom.txt").write_text("\n".join(lines) + "\n")
    print(f"report -> {RESULTS_DIR / 'probe_pool_geom.txt'}", flush=True)


if __name__ == "__main__":
    main()
