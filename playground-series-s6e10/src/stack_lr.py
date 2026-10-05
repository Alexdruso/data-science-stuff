"""PR-9: logistic-regression stack over the saved legs, with an honest outer CV.

    python src/stack_lr.py                 # all legs whose arrays exist
    python src/stack_lr.py --write         # also write the stacked test submission

Features: standardised logits of each leg's OOF (train) / test predictions.
The stacker is evaluated with an outer 5-fold CV over the OOF matrix (seed 7,
not the base folds' 42) and C is chosen inside each outer fold by an inner
3-fold CV, so the reported stack OOF is never scored on rows it was fit on.
Uses `l1_ratio=0` rather than penalty="l2" (sklearn 1.8 deprecation, see memory).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from blend_check import SUBMITTED_BLEND
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
LEGS = ["lgbm_pr5_tuned", "lgbm_pr1_base", "xgb_v1", "catboost_v1", "realmlp_v1", "tabm_v1"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
BAR = 0.0005


def _logits(a: np.ndarray) -> np.ndarray:
    return logit(np.clip(a, 1e-6, 1 - 1e-6))


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
    best_c, best = C_GRID[0], -1.0
    for c in C_GRID:
        scores = []
        for tr, va in inner.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=c, l1_ratio=0, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            scores.append(roc_auc_score(y[va], lr.decision_function(sc.transform(X[va]))))
        if np.mean(scores) > best:
            best_c, best = c, float(np.mean(scores))
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=best_c, l1_ratio=0, max_iter=1000).fit(sc.transform(X), y)


def main() -> None:
    legs = [n for n in LEGS if (RESULTS_DIR / f"oof_{n}.npy").exists()]
    print(f"legs: {legs}")
    # Row-order invariant: y and ids through build_features.
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    X = np.column_stack([_logits(np.load(RESULTS_DIR / f"oof_{n}.npy")) for n in legs])
    X_te = np.column_stack([_logits(np.load(RESULTS_DIR / f"test_{n}.npy")) for n in legs])

    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(X, y):
        sc, lr = _fit(X[tr], y[tr])
        oof[va] = lr.decision_function(sc.transform(X[va]))
    control = sum(
        w * rankdata(np.load(RESULTS_DIR / f"oof_{n}.npy")) for n, w in SUBMITTED_BLEND.items()
    )
    s_stack, s_ctrl = roc_auc_score(y, oof), roc_auc_score(y, control)
    sc, lr = _fit(X, y)
    print(f"final C={lr.C}  coefs (standardised logits): "
          f"{dict(zip(legs, np.round(lr.coef_[0], 3)))}")
    print(f"stack OOF {s_stack:.6f}   control {s_ctrl:.6f}")
    d = s_stack - s_ctrl
    print(f"DELTA {d:+.6f}  bar +{BAR}  → {'ADOPT' if d >= BAR else 'FAIL'}")
    np.save(RESULTS_DIR / "oof_stack_lr.npy", oof)

    if "--write" in sys.argv:
        test = lr.predict_proba(sc.transform(X_te))[:, 1]
        np.save(RESULTS_DIR / "test_stack_lr.npy", test)
        print(write_submission(SUBMISSIONS_DIR, "stack_lr.csv", test_ids, TARGET, test))


if __name__ == "__main__":
    main()
