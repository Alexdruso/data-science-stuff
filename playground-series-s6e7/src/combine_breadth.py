"""Track A combine — average the per-seed breadth arrays into the final #2 candidate.

`run_breadth.sh` writes one oof_<model>_s<seed>.npy per (model, seed) so a reboot only
loses the in-flight fit. This averages whatever seeds survived (skip-missing, so it's
runnable at any point) into oof_<model>_breadth.npy, then blends the 4 GBDT-family bases on
the decision-corrected surface (the v1b recipe: precorrect → Nelder-Mead → decision weights)
into submissions/final2_breadth.csv. Water is kept (features default) — champion recipe.

Reuses ensemble.precorrect/blend/objective + train_common.robust_decision_weights, so the
blend math is identical to the champion pipeline.
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

SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777]
MODELS = ["lgbm", "hgbc", "xgboost", "catboost"]


def combine_seeds(model: str, y: np.ndarray) -> str | None:
    """Average present per-seed arrays for `model`; save *_breadth arrays + weights. Key or None."""
    oofs, tests, used = [], [], []
    for s in SEEDS:
        op, tp = RESULTS_DIR / f"oof_{model}_s{s}.npy", RESULTS_DIR / f"test_{model}_s{s}.npy"
        if op.exists() and tp.exists():
            oofs.append(np.load(op))
            tests.append(np.load(tp))
            used.append(s)
    if not oofs:
        print(f"  {model}: no per-seed arrays yet — skip")
        return None
    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    key = f"{model}_breadth"
    np.save(RESULTS_DIR / f"oof_{key}.npy", oof)
    np.save(RESULTS_DIR / f"test_{key}.npy", test)
    dw = robust_decision_weights(y, oof)
    with (RESULTS_DIR / f"decision_weights_{key}.json").open("w") as f:
        json.dump(dict(zip(CLASSES, dw.tolist())), f, indent=2)
    score = balanced_accuracy_score(y, weighted_predict(oof, dw))
    print(f"  {model}: combined {len(used)} seeds {used} → weighted OOF {score:.4f}")
    return key


def main() -> None:
    ds = load_dataset()
    y, test_ids = ds.y, ds.test_ids

    print("Averaging per-seed breadth arrays:")
    keys = [k for m in MODELS if (k := combine_seeds(m, y)) is not None]
    if len(keys) < 2:
        raise SystemExit("need >=2 combined bases for the breadth blend")

    oof = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    oof, test = precorrect(oof, test)  # deployed surface
    names = list(keys)
    oof_list = [oof[k] for k in names]
    test_list = [test[k] for k in names]

    res = minimize(objective, np.ones(len(names)), args=(oof_list, y),
                   method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    w = np.exp(res.x) / np.exp(res.x).sum()
    print(f"\nbreadth blend weights: {dict(zip(names, w.round(4)))}")
    blended_oof = blend(res.x, oof_list)
    blended_test = blend(res.x, test_list)
    dw = robust_decision_weights(y, blended_oof)
    score = balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    print(f"final2_breadth OOF balanced_acc (decision-tuned): {score:.4f}")

    np.save(RESULTS_DIR / "oof_ensemble_breadth.npy", blended_oof)
    np.save(RESULTS_DIR / "test_ensemble_breadth.npy", blended_test)
    with (RESULTS_DIR / "decision_weights_ensemble_breadth.json").open("w") as f:
        json.dump({"blend": dict(zip(names, w.tolist())),
                   "decision": dict(zip(CLASSES, dw.tolist()))}, f, indent=2)
    labels = np.array(CLASSES)[weighted_predict(blended_test, dw)]
    out = SUBMISSIONS_DIR / "final2_breadth.csv"
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(out, index=False)
    print(f"submission → {out}  ({len(labels)} rows)  dist {dict(pd.Series(labels).value_counts())}")


if __name__ == "__main__":
    main()
