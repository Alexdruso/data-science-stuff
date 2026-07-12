"""Decision-weight generalization audit (Day-10, TE-eng #3).

The decision layer is worth ~+0.010 and its weights have ALWAYS been fit on
the full OOF and evaluated in-sample on the same OOF — its optimism has never
been measured. Here: per candidate, (a) full-fit NM weights scored in-sample,
(b) honest 5-fold cross-fit (weights on 4/5, scored on held-out 1/5), and
(c) the cross-fitted (log-mean) weight vector's in-sample score, i.e. what a
shrunk deployment would read. Optimism = (a) - (b); if > 0.0005 the deployed
artifacts switch to LOFO-averaged weights.

Run: python src/probe_dw_lofo.py   -> results/probe_dw_lofo.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

KEYS = ["lrstack_u3_r", "hgbc_te3_u_breadth", "realmlp_te13_u_breadth"]
N_FOLDS = 5


def _fold_fit(y, proba, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    w = np.asarray(tune_decision_weights(y[fit_idx], proba[fit_idx]))
    honest = balanced_accuracy_score(
        y[eval_idx], weighted_predict(proba[eval_idx], w)
    )
    return np.log(w), float(honest)


def main() -> None:
    ds = load_dataset()
    y = ds.y

    jobs, meta = [], []
    for key in KEYS:
        proba = np.load(RESULTS_DIR / f"oof_{key}.npy")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        for fit_idx, eval_idx in skf.split(y, y):
            jobs.append(delayed(_fold_fit)(y, proba, fit_idx, eval_idx))
            meta.append(key)
        jobs.append(
            delayed(_fold_fit)(y, proba, np.arange(len(y)), np.arange(len(y)))
        )
        meta.append(f"{key}__full")

    results = Parallel(n_jobs=min(8, len(jobs)))(jobs)

    lines = [f"decision-weight LOFO audit, {N_FOLDS}-fold honest cross-fit"]
    for key in KEYS:
        proba = np.load(RESULTS_DIR / f"oof_{key}.npy")
        fold_res = [r for r, m in zip(results, meta) if m == key]
        full_logw, full_in = next(
            r for r, m in zip(results, meta) if m == f"{key}__full"
        )
        honest = np.array([r[1] for r in fold_res])
        logws = np.stack([r[0] for r in fold_res])
        w_lofo = np.exp(logws.mean(axis=0))
        lofo_in = balanced_accuracy_score(y, weighted_predict(proba, w_lofo))
        spread = np.exp(logws.std(axis=0)).round(4)
        optimism = full_in - honest.mean()
        lines.append(
            f"\n{key}\n"
            f"  full-fit in-sample      {full_in:.5f}\n"
            f"  honest cross-fit        {honest.mean():.5f} "
            f"(sd {honest.std():.5f}, folds {np.round(honest, 5)})\n"
            f"  OPTIMISM                {optimism:+.5f}"
            f"  {'-> DEPLOY LOFO WEIGHTS' if optimism > 0.0005 else '(below 0.0005 gate)'}\n"
            f"  lofo-avg w in-sample    {lofo_in:.5f}  "
            f"(vs full {full_in:.5f}; fold-weight spread x/{spread})\n"
            f"  w_full {np.exp(full_logw).round(4)}  w_lofo {w_lofo.round(4)}"
        )

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_dw_lofo.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
