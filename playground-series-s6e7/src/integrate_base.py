"""Integrate a gated-survivor base into a candidate ensemble (final3_<key>.csv).

Only run this for a base that PASSED diag_mlp_transfer.py (clear adv-weighted lift + test-like
diversity). Blends the champion 3-GBDT core + the new base on the decision-corrected surface
(v1b recipe: precorrect → Nelder-Mead → decision weights) — the champion recipe, one base added.
Never touches champion_v1.csv. Usage: python src/integrate_base.py <key>  (e.g. mlp_la / lgbm_ext).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from data_science_stuff.kaggle_utils import weighted_predict
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR, blend, objective, precorrect
from features import CLASSES, TARGET
from train_common import load_dataset, robust_decision_weights

CORE = ["lgbm", "xgboost", "catboost"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: integrate_base.py <key>")
    key = sys.argv[1]
    names = [*CORE, key]
    ds = load_dataset()
    y, test_ids = ds.y, ds.test_ids

    oof = {m: np.load(RESULTS_DIR / f"oof_{m}.npy") for m in names}
    test = {m: np.load(RESULTS_DIR / f"test_{m}.npy") for m in names}
    oof, test = precorrect(oof, test)
    oof_list = [oof[m] for m in names]
    test_list = [test[m] for m in names]

    res = minimize(objective, np.ones(len(names)), args=(oof_list, y), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    w = np.exp(res.x) / np.exp(res.x).sum()
    blended_oof = blend(res.x, oof_list)
    blended_test = blend(res.x, test_list)
    dw = robust_decision_weights(y, blended_oof)
    score = balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    print(f"core+{key} blend weights: {dict(zip(names, w.round(4)))}")
    print(f"core+{key} OOF balanced_acc (decision-tuned): {score:.4f}")

    np.save(RESULTS_DIR / f"oof_ensemble_{key}.npy", blended_oof)
    np.save(RESULTS_DIR / f"test_ensemble_{key}.npy", blended_test)
    with (RESULTS_DIR / f"decision_weights_ensemble_{key}.json").open("w") as f:
        json.dump({"blend": dict(zip(names, w.tolist())),
                   "decision": dict(zip(CLASSES, dw.tolist()))}, f, indent=2)
    labels = np.array(CLASSES)[weighted_predict(blended_test, dw)]
    out = SUBMISSIONS_DIR / f"final3_{key}.csv"
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(out, index=False)
    print(f"submission → {out}  ({len(labels)} rows)  dist {dict(pd.Series(labels).value_counts())}")


if __name__ == "__main__":
    main()
