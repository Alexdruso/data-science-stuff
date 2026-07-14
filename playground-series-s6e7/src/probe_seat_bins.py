"""Bins seat test (Day-10/11): do the per-class specialist binaries earn a seat?

Rerun of the Day-10 inline probe whose stdout was lost to an overnight shutdown
(results/probe_seat_bins.txt held only joblib warnings). Same 12 split-half cells
as probe_combiner_ftt.py (LR meta + decision weights refit on half A, scored on
half B):

  lr8   — the lrstack_u3 leg set (incumbent shape)
  lrb   — lr8 + the two bins_te3 specialist columns (fit-vs-rest, unh-vs-rest)

bins arrays are (n, 2) = [P(fit), P(unhealthy)] from src/train_binaries_te3.py;
logit-transformed like every other leg before entering the LR design.
Expected flat per the 3-dim label-channel finding — closing the open loop.

Run: python src/probe_seat_bins.py -> results/probe_seat_bins.txt
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
BINS = "bins_te3_u_s42"


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


def cell(X8, Xb, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {
        "lr8": _arm(X8, y, fit_idx, eval_idx),
        "lrb": _arm(Xb, y, fit_idx, eval_idx),
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    }
    bins = clipped_logit(np.load(RESULTS_DIR / f"oof_{BINS}.npy")).astype(np.float32)
    X8 = np.hstack([blocks[k] for k in KEYS8])
    Xb = np.hstack([X8, bins])

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X8, Xb, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [f"bins seat test, {len(cells)} cells; lr8={KEYS8}, +{BINS} (2 cols)"]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_seat_bins.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
