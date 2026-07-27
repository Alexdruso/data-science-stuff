"""Day-19 — decision-weight stability check (fine-tuning pass, no retraining).

``robust_decision_weights`` (train_common.py) bags ``n_bags=8`` bootstrap
Nelder-Mead fits because a single NM fit has ~0.004 balanced-acc variance on
this OOF (its own docstring) — noise an order of magnitude larger than most
deltas this campaign has chased. Raising the bootstrap count is a pure
variance-reduction move on an already-computed OOF array (no GPU retraining,
no new measurement needed to justify it): more bootstrap resamples can only
tighten the weight estimate.

For every finals-menu key with a saved oof/test proba array, this recomputes
decision weights at n_bags=32 (up from the deployed 8) straight from the
cached OOF and reports how much the weights/predictions actually move. It
NEVER overwrites an existing submissions/*.csv — a new candidate file is only
written (distinct name, suffix ``_dw32``) if predictions actually change on
OOF; otherwise this is a stability confirmation, not a change.

Run: python src/audit_decision_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR
from features import CLASSES, TARGET
from train_common import load_dataset, robust_decision_weights

from data_science_stuff.kaggle_utils import weighted_predict

# Menu keys that follow the oof_<key>.npy / test_<key>.npy / decision_weights_<key>.json
# convention. champion_v1 is excluded: it's a precorrected blend with no single
# decision-weights JSON in this shape, so it doesn't fit this generic recompute.
KEYS = [
    "lrstack_u4_r",
    "lrstack_u3_r",
    "lrstack6b_r",
    "realmlp_te13_u_breadth",
    "hgbc_te3_u_breadth",
]

N_BAGS_NEW = 32


def main() -> None:
    ds = load_dataset()
    y = ds.y
    test_ids = ds.test_ids
    lines = ["decision-weight stability check (n_bags 8 -> 32, no retraining)"]

    for key in KEYS:
        oof = np.load(RESULTS_DIR / f"oof_{key}.npy")
        old_dwj = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
        old_w = (
            np.array(list(old_dwj["decision"].values()))
            if "decision" in old_dwj
            else np.array([old_dwj[c] for c in CLASSES])
        )
        new_w = robust_decision_weights(y, oof, n_bags=N_BAGS_NEW)

        old_pred = weighted_predict(oof, old_w)
        new_pred = weighted_predict(oof, new_w)
        flips = int((old_pred != new_pred).sum())
        from sklearn.metrics import balanced_accuracy_score

        old_ba = balanced_accuracy_score(y, old_pred)
        new_ba = balanced_accuracy_score(y, new_pred)
        w_delta = np.abs(new_w - old_w).max()

        lines.append(
            f"  {key:<26} old_w={np.round(old_w, 4).tolist()} "
            f"new_w={np.round(new_w, 4).tolist()} max|dw|={w_delta:.4f} "
            f"OOF flips={flips} ({flips / len(y):.5%}) "
            f"bacc old={old_ba:.5f} new={new_ba:.5f}"
        )

        if flips > 0:
            test_proba = np.load(RESULTS_DIR / f"test_{key}.npy")
            new_test_pred = weighted_predict(test_proba, new_w)
            out_key = f"{key}_dw32"
            np.save(RESULTS_DIR / f"decision_weights_{out_key}.npy", new_w)
            with (RESULTS_DIR / f"decision_weights_{out_key}.json").open("w") as f:
                json.dump(
                    {**dict(zip(CLASSES, new_w.tolist())), "n_bags": N_BAGS_NEW}, f
                )
            out_csv = SUBMISSIONS_DIR / f"{out_key}.csv"
            pd.DataFrame(
                {"id": test_ids, TARGET: np.array(CLASSES)[new_test_pred]}
            ).to_csv(out_csv, index=False)
            lines.append(
                f"    -> wrote new candidate {out_csv.name} (test flips vs old-weight prediction not measured here; run finals_audit.py on it)"
            )
        else:
            lines.append(
                "    -> weights/predictions unchanged at OOF resolution; no new file written"
            )

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "decision_weight_stability.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
