"""Blend an explicit list of base keys into a named candidate submission (v1b recipe).

Generic version of the champion blend: precorrect each base to its deployed decision surface,
Nelder-Mead the blend weights, tune decision weights on the blend. Used for final #2 (the
breadth-averaged 4 GBDT bases already on disk as *_bagw) and any ad-hoc candidate.

Usage: python src/blend_named.py <out_name> <key1> <key2> ...
   e.g. python src/blend_named.py final2_breadth lgbm_bagw xgboost_bagw catboost_bagw hgbc_bagw
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


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: blend_named.py <out_name> <key1> <key2> [...]")
    out_name, keys = sys.argv[1], sys.argv[2:]
    ds = load_dataset()
    y, test_ids = ds.y, ds.test_ids

    oof = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    for k in keys:
        s = balanced_accuracy_score(y, np.argmax(oof[k], axis=1))
        print(f"  base {k}: argmax OOF {s:.4f}")
    oof, test = precorrect(oof, test)
    oof_list = [oof[k] for k in keys]
    test_list = [test[k] for k in keys]

    res = minimize(objective, np.ones(len(keys)), args=(oof_list, y), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    w = np.exp(res.x) / np.exp(res.x).sum()
    blended_oof = blend(res.x, oof_list)
    blended_test = blend(res.x, test_list)
    dw = robust_decision_weights(y, blended_oof)
    score = balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    print(f"\n{out_name} blend weights: {dict(zip(keys, w.round(4)))}")
    print(f"{out_name} OOF balanced_acc (decision-tuned): {score:.4f}")

    np.save(RESULTS_DIR / f"oof_{out_name}.npy", blended_oof)
    np.save(RESULTS_DIR / f"test_{out_name}.npy", blended_test)
    with (RESULTS_DIR / f"decision_weights_{out_name}.json").open("w") as f:
        json.dump({"blend": dict(zip(keys, w.tolist())),
                   "decision": dict(zip(CLASSES, dw.tolist()))}, f, indent=2)
    labels = np.array(CLASSES)[weighted_predict(blended_test, dw)]
    out = SUBMISSIONS_DIR / f"{out_name}.csv"
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(out, index=False)
    print(f"submission → {out}  ({len(labels)} rows)  dist {dict(pd.Series(labels).value_counts())}")


if __name__ == "__main__":
    main()
