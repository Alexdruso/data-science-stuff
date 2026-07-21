"""Day-17 seat tests: cascade_te3_u and tabms_te13_u challengers.

The two Day-9 assertion-kills re-measured on the deployed recipe (te13, u
surface) — the FTT precedent says pre-TE vetoes must be re-earned. Standard
12-cell split-half instrument (probe_seat_bins/probe_seat_disp pattern):

  lr8   — incumbent lrstack leg set
  lrc   — lr8 + cascade_te3_u_s42 logits (target-factorization challenger)
  lrt   — lr8 + tabms_te13_u_s42 logits (BatchEnsemble architecture challenger)
  lrct  — lr8 + both

Pooled-veto MDE 6.5e-5 (Day-14): a null is a clean closure.

Run: python src/probe_seat_day17.py -> results/probe_seat_day17.txt
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
CHALLENGERS = {
    "lrc": ["cascade_te3_u_s42"],
    "lrt": ["tabms_te13_u_s42"],
    "lrct": ["cascade_te3_u_s42", "tabms_te13_u_s42"],
}


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


def cell(mats, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {name: _arm(X, y, fit_idx, eval_idx) for name, X in mats.items()}


def main() -> None:
    ds = load_dataset()
    y = ds.y

    def logits(key: str) -> np.ndarray:
        return clipped_logit(np.load(RESULTS_DIR / f"oof_{key}.npy")).astype(
            np.float32
        )

    X8 = np.hstack([logits(k) for k in KEYS8])
    mats = {"lr8": X8}
    for name, keys in CHALLENGERS.items():
        mats[name] = np.hstack([X8] + [logits(k) for k in keys])
    print({n: X.shape for n, X in mats.items()}, flush=True)

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(delayed(cell)(mats, y, f, e) for f, e in cells)
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [
        f"day-17 seat tests, {len(cells)} cells; lr8={KEYS8}",
        f"challengers: { {n: k for n, k in CHALLENGERS.items()} }",
    ]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    for a in CHALLENGERS:
        d = scores[a] - scores["lr8"]
        lines.append(f"per-cell {a}: " + " ".join(f"{x:+.5f}" for x in d))
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_seat_day17.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
