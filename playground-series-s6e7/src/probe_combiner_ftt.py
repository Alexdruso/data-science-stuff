"""FTT seat test (Day-10): does the 4th TE family earn an LR-stack seat?

Day-9's u-surface verdict was "top-2 legs carry the whole stack" — but every
challenger then was a clone of an incumbent family. ftt_te13 (0.9504) is a
genuinely different architecture fed the same recipe. Standard 12 split-half
cells (LR meta + decision weights refit on half A, scored on half B):

  lr8 — the lrstack_u3 leg set (incumbent shape)
  lr9 — lr8 + ftt_te13_u_s42
  lr3 — hgbc_te3 + realmlp_te13 + ftt_te13 breadths/seed (three-family core)

Run: python src/probe_combiner_ftt.py -> results/probe_combiner_ftt.txt
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
FTT = "ftt_te13_u_s42"
TOP3 = ["hgbc_te3_u_breadth", "realmlp_te13_u_breadth", FTT]


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


def cell(X8, X9, X3, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {
        "lr8": _arm(X8, y, fit_idx, eval_idx),
        "lr9": _arm(X9, y, fit_idx, eval_idx),
        "lr3": _arm(X3, y, fit_idx, eval_idx),
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in [*KEYS8, FTT]
    }
    X8 = np.hstack([blocks[k] for k in KEYS8])
    X9 = np.hstack([blocks[k] for k in [*KEYS8, FTT]])
    X3 = np.hstack([blocks[k] for k in TOP3])

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X8, X9, X3, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [f"FTT seat test, {len(cells)} cells; lr8={KEYS8}, +{FTT}"]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_combiner_ftt.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
