"""Build Day-3 attribution submissions for PS S6E7 (no Kaggle calls — CSVs only).

Each CSV isolates ONE change from the champion (v1b: single-seed 3-GBDT, water kept):
    attrib_bagw.csv   — bagged 3-GBDT, water KEPT   → isolates BAGGING
    attrib_bag.csv    — bagged 3-GBDT, water DROPPED → adds the water effect
    champion_v1.csv   — reconstructed v1b            → reproducibility check
All use the exact v1b ensemble recipe: per-model single-shot decision weights →
precorrect → NM blend on OOF → single-shot decision weights on blend → labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict
from features import CLASSES, TARGET
from lb_anchor import correct
from train_common import RESULTS_DIR, SUBMISSIONS_DIR, load_dataset

GBDTS = ["lgbm", "xgboost", "catboost"]


def v1b_recipe(
    suffix: str, y: NDArray, test_ids: NDArray, out_name: str
) -> float | None:
    """Apply the champion ensemble recipe to arrays with the given suffix; write CSV."""
    try:
        oof = {m: np.load(RESULTS_DIR / f"oof_{m}{suffix}.npy") for m in GBDTS}
        test = {m: np.load(RESULTS_DIR / f"test_{m}{suffix}.npy") for m in GBDTS}
    except FileNotFoundError as e:
        print(f"[{out_name}] arrays missing ({e}) — skipped")
        return None

    w = {m: tune_decision_weights(y, oof[m]) for m in GBDTS}
    pc_oof = [correct(oof[m], w[m]) for m in GBDTS]
    pc_test = [correct(test[m], w[m]) for m in GBDTS]

    def blend(wv: NDArray, arrays: list[NDArray]) -> NDArray:
        s = np.exp(wv) / np.exp(wv).sum()
        return sum(s[i] * a for i, a in enumerate(arrays))

    res = minimize(
        lambda wv: (
            -float(balanced_accuracy_score(y, np.argmax(blend(wv, pc_oof), axis=1)))
        ),
        np.ones(len(GBDTS)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    blended_oof = blend(res.x, pc_oof)
    dw = tune_decision_weights(y, blended_oof)
    oof_score = float(balanced_accuracy_score(y, weighted_predict(blended_oof, dw)))

    labels = np.array(CLASSES)[weighted_predict(blend(res.x, pc_test), dw)]
    out = SUBMISSIONS_DIR / f"{out_name}.csv"
    pl.DataFrame({"id": test_ids, TARGET: labels}).write_csv(out)
    s = np.exp(res.x) / np.exp(res.x).sum()
    print(
        f"[{out_name}] OOF={oof_score:.5f}  blend={dict(zip(GBDTS, s.round(3)))} → {out}"
    )
    return oof_score


def main() -> None:
    ds = load_dataset()

    # champion reproduction CSV from the saved labels (forensics_v1b_v2.py)
    champ = RESULTS_DIR / "champion_test_labels.npy"
    if champ.exists():
        labels = np.array(CLASSES)[np.load(champ)]
        out = SUBMISSIONS_DIR / "champion_v1.csv"
        pl.DataFrame({"id": ds.test_ids, TARGET: labels}).write_csv(out)
        print(f"[champion_v1] reconstructed v1b labels → {out}")

    v1b_recipe("_bagw", ds.y, ds.test_ids, "attrib_bagw")  # bagging only
    v1b_recipe("_bag", ds.y, ds.test_ids, "attrib_bag")  # bagging + water drop


if __name__ == "__main__":
    main()
