"""Day-10 PM seat tests: ncw (unweighted posteriors) + knn (joint-density).

Both candidates are solo-flat (0.9503/0.9504 vs 0.9504) but differ by
MECHANISM, not architecture: ncw's posteriors carry a different within-band
bias (no 15x minority upweighting — the mlp_la_r precedent: a per-class meta
extracted value scalar NM couldn't); knn adds a statistic no incumbent leg
computes. Standard 12 split-half cells vs the lr8 incumbent leg set.

Run: python src/probe_combiner_pm.py -> results/probe_combiner_pm.txt
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
NCW = "hgbc_te3ncw_u_s42"
KNN = "hgbc_te3knn_u_s42"


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


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in [*KEYS8, NCW, KNN]
    }
    designs = {
        "lr8": np.hstack([blocks[k] for k in KEYS8]),
        "ncw": np.hstack([blocks[k] for k in [*KEYS8, NCW]]),
        "knn": np.hstack([blocks[k] for k in [*KEYS8, KNN]]),
        "both": np.hstack([blocks[k] for k in [*KEYS8, NCW, KNN]]),
    }

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    def cell(fit_idx, eval_idx):  # noqa: ANN001, ANN202
        return {name: _arm(X, y, fit_idx, eval_idx) for name, X in designs.items()}

    results = Parallel(n_jobs=4)(delayed(cell)(f, e) for f, e in cells)
    scores = {a: np.array([r[a] for r in results]) for a in designs}
    lines = [f"Day-10 PM seat tests, {len(cells)} cells; +{NCW}, +{KNN}"]
    for a in sorted(designs, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<5} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_combiner_pm.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
