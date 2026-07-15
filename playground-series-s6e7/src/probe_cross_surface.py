"""Day-13 Rung 1 — cross-surface combination read on the intersection instrument.

Question: does AVERAGING the two-surface finalists (lrstack_u3, plain-trained;
lrstack_rte13_r, repaired-trained) beat both parents where their val inputs are
identical? All 8 prior seat vetoes tested same-surface legs; the u-vs-r surface
difference is the one decorrelation left with mass (it's what the finals hedge
monetizes). If the average pays here, Rung 2 builds surface-consistent `_rt`
legs (fit-on-remasked, predict-on-plain) for a valid joint stack.

Instrument: rows complete in the 4 MECHANISM_SHIFTED columns under the m100
remask stream — on those rows the repaired lineage's val inputs equal the plain
ones, so u- and r-lineage OOFs are directly comparable (read_lineage mechanism).
Both-streams intersection also reported to cross-check the Day-10 recorded
numbers (rte13_r 0.95156 / u3 0.95172).

Scoring: per candidate, robust_decision_weights refit ON the subset (same fixed
RNG -> paired across candidates; the decision layer's in-sample optimism is
measured <= +0.0002 and identical across arms), then subset bacc. A w-grid over
the r-share probes whether any mixture beats the better parent.

Gate for Rung 2: best mixture >= +0.0002 over the better parent on the m100
subset.

Run: python src/probe_cross_surface.py   (no S6E7_REPAIR)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES

from data_science_stuff.kaggle_utils import weighted_predict

assert os.environ.get("S6E7_REPAIR", "") in ("", "0"), "run without S6E7_REPAIR"
from train_common import (  # noqa: E402 — env assert must precede import-time REPAIR
    MECHANISM_SHIFTED,
    RESULTS_DIR,
    _uniform_remask,
    load_dataset,
    robust_decision_weights,
)

U_KEY = "lrstack_u3_r"
R_KEY = "lrstack_rte13_r"
R_SHARES = [0.0, 0.2, 0.35, 0.5, 0.65, 1.0]


def saved_weights(key: str) -> np.ndarray:
    dwj = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
    return (
        np.array(list(dwj["decision"].values()))
        if "decision" in dwj
        else np.array([dwj[c] for c in CLASSES])
    )


def complete4_under(mult: str, ds) -> np.ndarray:  # noqa: ANN001
    os.environ["S6E7_REPAIR_MULT"] = mult
    remasked = _uniform_remask(ds.train, ds.test)
    cols = [c for c in MECHANISM_SHIFTED if c in ds.train.columns]
    return remasked[cols].notna().all(axis=1).to_numpy()


def main() -> None:
    ds = load_dataset()
    y = ds.y
    c4_m100 = complete4_under("1.0", ds)
    inter2 = c4_m100 & complete4_under("0.5", ds)
    print(f"m100-complete subset: {c4_m100.sum():,} rows ({c4_m100.mean():.1%})")
    print(f"both-streams intersection: {inter2.sum():,} rows ({inter2.mean():.1%})")

    oof_u = np.load(RESULTS_DIR / f"oof_{U_KEY}.npy")
    oof_r = np.load(RESULTS_DIR / f"oof_{R_KEY}.npy")

    # sanity vs Day-10 recorded numbers (saved weights, both-streams subset)
    print("\n--- sanity (saved weights) ---")
    for key, oof in [(U_KEY, oof_u), (R_KEY, oof_r)]:
        pred = weighted_predict(oof, saved_weights(key))
        print(
            f"{key:<18} inter2={bacc(y[inter2], pred[inter2]):.5f} "
            f"m100={bacc(y[c4_m100], pred[c4_m100]):.5f} all={bacc(y, pred):.5f}"
        )

    # paired mixture sweep, weights refit on the subset per arm
    for sub_name, sub in [("m100-complete", c4_m100), ("both-streams", inter2)]:
        print(f"\n--- mixture sweep on {sub_name} (refit weights, paired) ---")
        for w_r in R_SHARES:
            mix = (1.0 - w_r) * oof_u + w_r * oof_r
            dw = robust_decision_weights(y[sub], mix[sub])
            pred = weighted_predict(mix[sub], dw)
            print(
                f"w_r={w_r:.2f}  bacc={bacc(y[sub], pred):.5f}  "
                f"dw={np.round(dw, 3).tolist()}"
            )

    # test-side disagreement diagnostics (saved weights — the deployed artifacts)
    print("\n--- test-side disagreement (deployed labels) ---")
    test_u = np.load(RESULTS_DIR / f"test_{U_KEY}.npy")
    test_r = np.load(RESULTS_DIR / f"test_{R_KEY}.npy")
    lab_u = weighted_predict(test_u, saved_weights(U_KEY))
    lab_r = weighted_predict(test_r, saved_weights(R_KEY))
    dis = lab_u != lab_r
    cols = [c for c in MECHANISM_SHIFTED if c in ds.test.columns]
    nan4 = ds.test[cols].isna().any(axis=1).to_numpy()
    print(
        f"disagree: {dis.sum():,}/{len(dis):,} ({dis.mean():.3%}); "
        f"shifted-col-NaN share on disagreements {nan4[dis].mean():.1%} "
        f"vs base {nan4.mean():.1%}"
    )
    print("class-move matrix (rows=u3 label, cols=rte13_r label):")
    mv = np.zeros((3, 3), dtype=int)
    for a, b in zip(lab_u[dis], lab_r[dis]):
        mv[a, b] += 1
    for i, cname in enumerate(CLASSES):
        print(f"  {cname:<10} {mv[i].tolist()}")


if __name__ == "__main__":
    main()
