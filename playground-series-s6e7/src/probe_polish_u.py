"""Day-13 PM — direct-metric meta polish (advisor idea #1) + power calibration (#2).

Mechanism #1: the LR meta minimizes weighted log-loss; the metric is balanced
accuracy (argmax placement only). Starting FROM the honest LR solution, NM-
polish a 9-param reparametrization directly on dw-bacc: per-class additive
gains on the top-2 legs' logit columns + per-class intercepts. No documented
kill covers metric-fitting the meta's own coefficients (NM kill = scalar-per-
model; cost-matrix kill = post-meta linear map; decision weights = per-class
scale only).

Mechanism #2: decision rule argmax(w_c*p_c) assumes reliability is right up to
one scalar per class; argmax(w_c*p_c^alpha_c) lets the tilt vary with
confidence (the at-risk<->minority band is all low-confidence). Outside LR's
hypothesis class (power on probs != linear map on logits after softmax... it
IS logit scaling per class — but per-class TEMPERATURE was never measured
either; both arms are cheap on the same harness).

Same 12 split-half cells: everything (LR, polish, alpha, weights) fit on half
A, scored once on half B. Adoption bar >= +0.0003 at >=10/12.

Run: python src/probe_polish_u.py -> results/probe_polish_u.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
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
TOP2 = ["hgbc_te3_u_breadth", "realmlp_te13_u_breadth"]


def _bacc_of_scores(scores: np.ndarray, y: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, scores.argmax(axis=1)))


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def cell(X, top2_cols, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    out = {}
    pf, pe = lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])
    out["lr"] = _decide(y[fit_idx], pf, pe, y[eval_idx])

    # decision scores (pre-softmax) on both halves
    sf = X[fit_idx] @ lr.coef_.T + lr.intercept_
    se = X[eval_idx] @ lr.coef_.T + lr.intercept_
    # top-2 legs' per-class logit columns
    tf = X[fit_idx][:, top2_cols]  # (n, 6): [leg1_c0..c2, leg2_c0..c2]
    te = X[eval_idx][:, top2_cols]

    # dw-bacc baseline in score space on the FIT half (weights ~ intercepts here)
    def polish_obj(theta: np.ndarray) -> float:
        g1, g2, b = theta[0:3], theta[3:6], theta[6:9]
        s = sf + tf[:, 0:3] * g1 + tf[:, 3:6] * g2 + b
        return -_bacc_of_scores(s, y[fit_idx])

    res = minimize(
        polish_obj, np.zeros(9), method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-4, "fatol": 1e-6},
    )
    g1, g2, b = res.x[0:3], res.x[3:6], res.x[6:9]
    out["polish9"] = _bacc_of_scores(
        se + te[:, 0:3] * g1 + te[:, 3:6] * g2 + b, y[eval_idx]
    )

    # per-class temperature on LR probs: argmax(w_c * p_c^a_c); fit (a, log w) on A
    lpf, lpe = np.log(np.clip(pf, 1e-9, 1)), np.log(np.clip(pe, 1e-9, 1))

    def alpha_obj(theta: np.ndarray) -> float:
        a, lw = theta[0:3], theta[3:6]
        return -_bacc_of_scores(lpf * a + lw, y[fit_idx])

    res2 = minimize(
        alpha_obj, np.concatenate([np.ones(3), np.zeros(3)]),
        method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-6},
    )
    a, lw = res2.x[0:3], res2.x[3:6]
    out["alpha_c"] = _bacc_of_scores(lpe * a + lw, y[eval_idx])
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    }
    X = np.hstack([blocks[k] for k in KEYS8])
    top2_cols = []
    for k in TOP2:
        j = KEYS8.index(k) * 3
        top2_cols += [j, j + 1, j + 2]

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X, top2_cols, y, f, e) for f, e in cells
    )
    arm_names = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arm_names}
    lines = [f"direct-metric meta polish, {len(cells)} cells; top2={TOP2}"]
    for a in sorted(arm_names, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr"]
        lines.append(
            f"  {a:<8} {scores[a].mean():.5f}   vs lr {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_polish_u.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
