"""Build the unrepaired-surface submission candidates (Day-9 final block, E3).

From the 719199 artifacts:
  1. hgbc_te3_u_breadth.csv — 3-seed avg of the v0.7 replica (0.9504)
  2. realmlp_u_breadth.csv  — 3-seed avg of unrepaired realmlp (0.9503)
  3. blend2_u.csv           — NM blend of {1,2} on the decision-corrected
                              surface (Susol v0.9 shape, his best LB)
  4. ublend_nm.csv          — NM blend of all 7 unrepaired legs (final3-analog)
  5. metablend_u.csv        — avg(lrstack_u, ublend_nm) + decision weights
                              (the s6e6 best-LB shape on the new surface)

Blend/metablend OOF numbers are full-fit NM (optimistic) — these are
submission artifacts, not gate reads; the honest number is lrstack_u's 0.9506.
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
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR, blend, objective, precorrect
from features import CLASSES, TARGET
from train_common import load_dataset, robust_decision_weights

from data_science_stuff.kaggle_utils import weighted_predict

KEYS7 = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "hgbc_te3_u_breadth",
    "realmlp_u_breadth",
    "mlp",
]


def write_sub(name: str, test_proba: np.ndarray, w: np.ndarray, test_ids) -> None:  # noqa: ANN001
    labels = np.array(CLASSES)[weighted_predict(test_proba, w)]
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(
        SUBMISSIONS_DIR / f"{name}.csv", index=False
    )
    print(f"  -> submissions/{name}.csv")


def nm_blend(y, oof_pc: list[np.ndarray], test_pc: list[np.ndarray]):  # noqa: ANN001, ANN202
    res = minimize(
        objective,
        np.ones(len(oof_pc)),
        args=(oof_pc, y),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
    )
    s = np.exp(res.x) / np.exp(res.x).sum()
    return blend(res.x, oof_pc), blend(res.x, test_pc), s


def main() -> None:
    ds = load_dataset()
    y = ds.y
    oof_raw = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in KEYS7}
    test_raw = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in KEYS7}
    oof_pc, test_pc = precorrect(oof_raw, test_raw)

    # 1+2: breadth solos
    for key in ("hgbc_te3_u_breadth", "realmlp_u_breadth"):
        wd = json.loads((RESULTS_DIR / f"decision_weights_{key}.json").read_text())
        w = np.array([wd[c] for c in CLASSES])
        score = balanced_accuracy_score(y, weighted_predict(oof_raw[key], w))
        print(f"[{key}] weighted OOF {score:.4f}")
        write_sub(key, test_raw[key], w, ds.test_ids)

    # 3: v0.9-shape 2-leg NM blend
    pair = ["hgbc_te3_u_breadth", "realmlp_u_breadth"]
    b_oof, b_test, s = nm_blend(y, [oof_pc[k] for k in pair], [test_pc[k] for k in pair])
    w = robust_decision_weights(y, b_oof)
    score = balanced_accuracy_score(y, weighted_predict(b_oof, w))
    print(f"[blend2_u] NM weights {dict(zip(pair, s.round(3)))} weighted OOF {score:.4f}")
    write_sub("blend2_u", b_test, w, ds.test_ids)

    # 4: all-7 NM blend
    b7_oof, b7_test, s7 = nm_blend(
        y, [oof_pc[k] for k in KEYS7], [test_pc[k] for k in KEYS7]
    )
    w7 = robust_decision_weights(y, b7_oof)
    score = balanced_accuracy_score(y, weighted_predict(b7_oof, w7))
    print(f"[ublend_nm] NM weights {dict(zip(KEYS7, s7.round(3)))} weighted OOF {score:.4f}")
    write_sub("ublend_nm", b7_test, w7, ds.test_ids)
    np.save(RESULTS_DIR / "oof_ublend_nm.npy", b7_oof)
    np.save(RESULTS_DIR / "test_ublend_nm.npy", b7_test)
    with (RESULTS_DIR / "decision_weights_ublend_nm.json").open("w") as f:
        json.dump(dict(zip(CLASSES, w7.tolist())), f, indent=2)

    # 5: metablend_u = avg(lrstack_u, ublend_nm) + decision weights
    lr_oof = np.load(RESULTS_DIR / "oof_lrstack_u_r.npy")
    lr_test = np.load(RESULTS_DIR / "test_lrstack_u_r.npy")
    mb_oof = (lr_oof + b7_oof) / 2
    mb_test = (lr_test + b7_test) / 2
    wm = robust_decision_weights(y, mb_oof)
    score = balanced_accuracy_score(y, weighted_predict(mb_oof, wm))
    print(f"[metablend_u] weighted OOF {score:.4f}")
    np.save(RESULTS_DIR / "oof_metablend_u.npy", mb_oof)
    np.save(RESULTS_DIR / "test_metablend_u.npy", mb_test)
    with (RESULTS_DIR / "decision_weights_metablend_u.json").open("w") as f:
        json.dump(dict(zip(CLASSES, wm.tolist())), f, indent=2)
    write_sub("metablend_u", mb_test, wm, ds.test_ids)


if __name__ == "__main__":
    main()
