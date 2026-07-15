"""Day-13 PM — the surface-diversity seat test (seat question #9, measured).

The Rung-1 mixture read killed cross-surface PROB-AVERAGING; this closes the
remaining consumer: does a per-class LR meta extract anything from
repaired-TRAINED legs whose OOF sits on the PLAIN surface (S6E7_REPAIR_FIT=1
`_rt` legs — the first repaired-trained artifacts that are surface-consistent
with the u legs, so joint meta-fitting is honest)?

Same 12 split-half cells as every seat test (LR meta + decision weights refit
on half A, scored on half B):

  lr8    — the lrstack_u3/u4 leg set (incumbent shape, 8-seed breadths)
  lrrt1  — lr8 + hgbc_te3_rt_u_s42
  lrrt2  — lr8 + realmlp_te13_rt_u_s42
  lrrt   — lr8 + both

Prior: veto (8 consecutive seat vetoes incl. better decorrelation profiles;
u3-vs-rte13 deployment disagreement 400 rows). Running it converts the bound
into a measurement while the box is otherwise idle.

Run: python src/probe_seat_rt.py -> results/probe_seat_rt.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
KEYS8 = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "mlp",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
    "realmlp_te13_u_breadth",
]
RT1 = "hgbc_te3_rt_u_s42"
RT2 = "realmlp_te13_rt_u_s42"


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _arm(X, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return _decide(
        y[fit_idx], lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx]),
        y[eval_idx],
    )


def cell(arms, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {name: _arm(X, y, fit_idx, eval_idx) for name, X in arms.items()}


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in [*KEYS8, RT1, RT2]
    }
    X8 = np.hstack([blocks[k] for k in KEYS8])
    arms = {
        "lr8": X8,
        "lrrt1": np.hstack([X8, blocks[RT1]]),
        "lrrt2": np.hstack([X8, blocks[RT2]]),
        "lrrt": np.hstack([X8, blocks[RT1], blocks[RT2]]),
    }

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(delayed(cell)(arms, y, f, e) for f, e in cells)
    arm_names = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arm_names}
    lines = [
        f"rt (surface-diversity) seat test, {len(cells)} cells; "
        f"lr8={KEYS8}, +{RT1}/+{RT2}"
    ]
    for a in sorted(arm_names, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<6} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_seat_rt.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
