"""LR-on-logits stacker over the repaired (m100) bases — s6e6's winning combiner.

s6e6 record: multinomial LR over per-(model,class) logit columns beat the scalar
Nelder-Mead blend 0.9698 vs 0.9657 once a strong diverse base joined the pool.
That precondition arrived in s6e7 today (realmlp). NM's structural limit is one
scalar per model; LR gets a weight per (model, class).

Reuses `data_science_stuff.kaggle.stacking.stack_oof` (honest multi-seed outer
5-fold cross-fit; logit transform internal). Two input variants:
  raw — base OOF probabilities as saved (stack_oof logits them)
  pc  — precorrected probabilities (`ensemble.precorrect`, the surface NM blends on)
Decision layer: `robust_decision_weights` on the stacked OOF (s6e7 convention).

Run: S6E7_REPAIR=1 python src/build_lr_stack.py [key1 key2 ...]
Default keys: the strong five (4 GBDT _r_breadth + realmlp_r_breadth).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR, precorrect
from features import CLASSES, TARGET
from train_common import MECHANISM_SHIFTED, load_dataset, robust_decision_weights

from data_science_stuff.kaggle.stacking import stack_oof
from data_science_stuff.kaggle_utils import weighted_predict

DEFAULT_KEYS = [
    "lgbm_r_breadth",
    "hgbc_r_breadth",
    "xgboost_r_breadth",
    "catboost_r_breadth",
    "realmlp_r_breadth",
]


def meta_factory() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, n_jobs=-1)


def main() -> None:
    keys = sys.argv[1:] or DEFAULT_KEYS
    ds = load_dataset()
    y = ds.y
    c4 = (
        ds.train[[c for c in MECHANISM_SHIFTED if c in ds.train.columns]]
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    oof_raw = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test_raw = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    oof_pc, test_pc = precorrect(dict(oof_raw), dict(test_raw))

    base_name = os.environ.get("S6E7_STACK_NAME", "lrstack")
    variants = os.environ.get("S6E7_STACK_VARIANTS", "raw,pc").split(",")
    all_variants = [("raw", oof_raw, test_raw), ("pc", oof_pc, test_pc)]
    for tag, oof_d, test_d in [v for v in all_variants if v[0] in variants]:
        name = f"{base_name}_r" if tag == "raw" else f"{base_name}_pc_r"
        s_oof, s_test = stack_oof(
            [oof_d[k] for k in keys],
            [test_d[k] for k in keys],
            y,
            meta_factory,
            use_logits=True,
        )
        dw = robust_decision_weights(y, s_oof)
        pred = weighted_predict(s_oof, dw)
        w_all = balanced_accuracy_score(y, pred)
        w_c4 = balanced_accuracy_score(y[c4], pred[c4])
        w_m4 = balanced_accuracy_score(y[~c4], pred[~c4])
        argmax = balanced_accuracy_score(y, s_oof.argmax(1))
        print(
            f"[{name}] argmax={argmax:.4f} weighted={w_all:.4f} "
            f"complete4={w_c4:.4f} miss4={w_m4:.4f}"
        )
        np.save(RESULTS_DIR / f"oof_{name}.npy", s_oof)
        np.save(RESULTS_DIR / f"test_{name}.npy", s_test)
        with (RESULTS_DIR / f"decision_weights_{name}.json").open("w") as f:
            json.dump(dict(zip(CLASSES, dw.tolist())), f, indent=2)
        labels = np.array(CLASSES)[weighted_predict(s_test, dw)]
        pd.DataFrame({"id": ds.test_ids, TARGET: labels}).to_csv(
            SUBMISSIONS_DIR / f"{name}.csv", index=False
        )
        print(f"  saved arrays + submissions/{name}.csv")


if __name__ == "__main__":
    main()
