"""Leg-swap probe (Day-12 PM): catboost_bagw -> catboost_te3_u_breadth in lr8.

catboost_te3 (Day-12: te13 through the 4th library, 0.9501/0.9500/0.9500 per
seed vs catboost_bagw 0.9492) upgrades the stack's weakest GBDT leg by +0.0009
solo. probe_combiner_u said lr = lr2 (top-2 legs carry everything), so the
expected verdict is a wash — but a leg UPGRADE (same information channel,
strictly stronger vehicle) is a different move from the seven vetoed leg
ADDITIONS, and it is the natural consumer read for today's new inventory.

Arms over the standard 12 split-half cells (probe_seat_bins.py protocol):
  lr8   — incumbent lrstack_u3 leg set
  lrsw  — same, catboost_bagw replaced by catboost_te3_u_breadth

Displacement bar for a stack rebuild: >= +0.0003 mean, >= 9/12 positive.

Run: python src/probe_swap_cat.py -> results/probe_swap_cat.txt
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
SWAP = {"catboost_bagw": "catboost_te3_u_breadth"}


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


def cell(X8, Xs, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {
        "lr8": _arm(X8, y, fit_idx, eval_idx),
        "lrsw": _arm(Xs, y, fit_idx, eval_idx),
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    keys_sw = [SWAP.get(k, k) for k in KEYS8]
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in set(KEYS8) | set(keys_sw)
    }
    X8 = np.hstack([blocks[k] for k in KEYS8])
    Xs = np.hstack([blocks[k] for k in keys_sw])

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X8, Xs, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [f"cat leg-swap test, {len(cells)} cells; swap {SWAP}"]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_swap_cat.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
