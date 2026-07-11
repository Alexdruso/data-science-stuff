"""Combiner verdict on the UNREPAIRED surface (Day-9 overtime, O3).

All combiner evidence to date (LR beats NM +0.0004; metablend average helps) is
repaired-surface. The finals will be assembled from the u-legs, so re-run the
honest instrument there: 12 split-half cells, every combiner + decision weights
refit on half A, scored on half B.

Arms:
  nm    — precorrect + Nelder-Mead over the 7 u-legs (ublend_nm shape)
  lr    — LR on per-(leg,class) logits, 7 legs (lrstack_u2 shape)
  lr2   — LR over ONLY the top-2 legs (hgbc_te3 + realmlp_te13; blend2 shape)
  mb    — avg(lr, nm) + decision weights (metablend_u shape)

Verdict picks WHICH assembled artifact is final #2 tomorrow.

Run: python src/probe_combiner_u.py   (no repair env — unrepaired surface)
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
from ensemble import RESULTS_DIR, blend, objective, precorrect
from train_common import load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
KEYS = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "hgbc_te3_u_breadth",
    "realmlp_te13_u_breadth",
    "mlp",
]
TOP2 = ["hgbc_te3_u_breadth", "realmlp_te13_u_breadth"]


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _fit_lr(blocks, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    X_fit = np.hstack([b[fit_idx] for b in blocks])
    X_eval = np.hstack([b[eval_idx] for b in blocks])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X_fit, y[fit_idx])
    return lr.predict_proba(X_fit), lr.predict_proba(X_eval)


def cell(oof_pc, blocks7, blocks2, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    pairs = {}
    fit_pc = [o[fit_idx] for o in oof_pc]
    res = minimize(
        objective,
        np.ones(len(oof_pc)),
        args=(fit_pc, y[fit_idx]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
    )
    pairs["nm"] = (blend(res.x, fit_pc), blend(res.x, [o[eval_idx] for o in oof_pc]))
    pairs["lr"] = _fit_lr(blocks7, y, fit_idx, eval_idx)
    pairs["lr2"] = _fit_lr(blocks2, y, fit_idx, eval_idx)
    pairs["mb"] = (
        np.mean([pairs["lr"][0], pairs["nm"][0]], axis=0),
        np.mean([pairs["lr"][1], pairs["nm"][1]], axis=0),
    )
    return {
        name: _decide(y[fit_idx], pf, pe, y[eval_idx])
        for name, (pf, pe) in pairs.items()
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    oof_raw = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in KEYS}
    test_raw = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in KEYS}
    oof_pc_d, _ = precorrect(dict(oof_raw), dict(test_raw))
    oof_pc = [oof_pc_d[k] for k in KEYS]
    blocks7 = [clipped_logit(oof_raw[k]) for k in KEYS]
    blocks2 = [clipped_logit(oof_raw[k]) for k in TOP2]

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(oof_pc, blocks7, blocks2, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    print(f"u-surface combiner tournament over {KEYS}; {len(cells)} cells")
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr"]
        print(
            f"  {a:<5} {scores[a].mean():.5f}   vs lr {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )


if __name__ == "__main__":
    main()
