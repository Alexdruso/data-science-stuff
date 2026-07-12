"""Regime-conditioned meta probe (Day-10, TE-eng #2) — pure OOF math.

Hypothesis: the LR stacker's per-(leg,class) weights should differ by
driver-missingness regime (models are differentially trustworthy where
sleep/stress/activity are masked). NOT the dead raw-features-in-meta (no
feature VALUES enter) and NOT the dead per-region decision weights (this is
upstream, on probabilities, with LR shrinkage toward the global solution).

Design: per-(leg,class) logits (24 cols) + 8 driver-missingness-pattern
dummies + logit x pattern interactions (192 cols). C sweeps the shrinkage
toward the plain stacker. Judged by the standard 12 split-half cells (meta +
decision weights refit on half A, scored on half B) vs the plain-lr arm over
the same u3 leg set.

Run: python src/probe_regime_meta.py   -> results/probe_regime_meta.txt
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
KEYS = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "mlp",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
    "realmlp_te13_u_breadth",
]
DRIVERS = ["sleep_duration", "stress_level", "physical_activity_level"]
C_GRID = [0.03, 0.3, 3.0]


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _fit(X, y, fit_idx, eval_idx, c):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])


def cell(X_plain, X_reg, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    out = {}
    pf, pe = _fit(X_plain, y, fit_idx, eval_idx, 1.0)
    out["lr"] = _decide(y[fit_idx], pf, pe, y[eval_idx])
    for c in C_GRID:
        pf, pe = _fit(X_reg, y, fit_idx, eval_idx, c)
        out[f"rm_C{c}"] = _decide(y[fit_idx], pf, pe, y[eval_idx])
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    logits = np.hstack(
        [clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")) for k in KEYS]
    ).astype(np.float32)

    miss = ds.train[DRIVERS].isna().to_numpy()
    pattern = miss @ (2 ** np.arange(len(DRIVERS)))
    P = np.eye(8, dtype=np.float32)[pattern]
    inter = (logits[:, :, None] * P[:, None, :]).reshape(len(y), -1)
    X_reg = np.hstack([logits, P, inter])
    print(
        f"{len(KEYS)} legs -> plain {logits.shape[1]} cols, "
        f"regime {X_reg.shape[1]} cols; pattern counts {np.bincount(pattern, minlength=8)}"
    )

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=3)(
        delayed(cell)(logits, X_reg, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [f"regime-conditioned meta, {len(cells)} split-half cells, legs {KEYS}"]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr"]
        lines.append(
            f"  {a:<8} {scores[a].mean():.5f}   vs lr {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_regime_meta.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
