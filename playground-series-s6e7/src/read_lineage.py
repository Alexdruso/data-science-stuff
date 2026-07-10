"""Cross-lineage paired read on the intersection subset.

The m100 (`_r`) and m050 (`_r2`) lineages train and validate on different
remasked surfaces, so their OOFs are not globally comparable. But on rows
COMPLETE in the 4 mechanism-shifted columns under BOTH remask streams, the val
inputs are identical — the deployed-lineage analog of the diag chain's TESTVOL.
This script scores any list of candidate keys (each with its decision-weights
JSON) on that intersection, plus each candidate's own-surface overall for the
record.

Run: python src/read_lineage.py <key1> <key2> ...  (no S6E7_REPAIR needed;
     both remask streams are applied internally to the raw matrix)
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
)


def complete4_under(mult: str, ds) -> np.ndarray:  # noqa: ANN001
    os.environ["S6E7_REPAIR_MULT"] = mult
    remasked = _uniform_remask(ds.train, ds.test)
    cols = [c for c in MECHANISM_SHIFTED if c in ds.train.columns]
    return remasked[cols].notna().all(axis=1).to_numpy()


def main() -> None:
    keys = sys.argv[1:]
    if not keys:
        raise SystemExit("usage: read_lineage.py <key1> <key2> ...")
    ds = load_dataset()  # raw
    inter = complete4_under("1.0", ds) & complete4_under("0.5", ds)
    print(
        f"intersection subset (complete-in-4 under BOTH streams): "
        f"{inter.sum():,} rows ({inter.mean():.1%})"
    )
    print(f"{'key':<26} {'own-surface all':>16} {'intersection':>13}")
    for k in keys:
        oof = np.load(RESULTS_DIR / f"oof_{k}.npy")
        dwj = json.load(open(RESULTS_DIR / f"decision_weights_{k}.json"))
        dw = (
            np.array(list(dwj["decision"].values()))
            if "decision" in dwj
            else np.array([dwj[c] for c in CLASSES])
        )
        pred = weighted_predict(oof, dw)
        print(
            f"{k:<26} {bacc(ds.y, pred):>16.4f} {bacc(ds.y[inter], pred[inter]):>13.4f}"
        )


if __name__ == "__main__":
    main()
