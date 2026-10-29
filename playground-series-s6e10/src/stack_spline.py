"""PR-45: spline-augmented LR stacker — the pr42 single LR (standardised members, C 0.01, nest 10)
plus hinge features max(0, z - q) of its K most-weighted members at their own quantiles, so each
heavy member enters through an additive piecewise-linear transform instead of a straight line.

    python src/stack_spline.py --ours $(cat results/ours_all80_*.txt) --top 40 --knots 0.2 0.4 0.6 0.8

Top-K is chosen by |coef| of one full-data LR (a mild selection on all rows; the coefficients
themselves are refit inside every nesting fold). Prints the no-hinge control first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_geometry import fold_fit
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def nested(X, XT, y, C, nest, seed, jobs):
    folds = list(StratifiedKFold(nest, shuffle=True, random_state=seed).split(X, y))
    res = Parallel(n_jobs=jobs)(delayed(fold_fit)(X, y, tr, va, XT, C, nest) for tr, va in folds)
    oof, test = np.zeros(len(y)), np.zeros(len(XT))
    for va, ov, tv in res:
        oof[va] = ov
        test += tv
    return oof, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--nest", type=int, default=10)
    p.add_argument("--nest-seed", type=int, default=42)
    p.add_argument("--top", type=int, nargs="+", default=[40])
    p.add_argument("--knots", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, T = load_pairs(a.ours, [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    mu, sd = O.mean(0), O.std(0)
    X, XT = (O - mu) / sd, (T - mu) / sd
    del O, T
    print(f"{len(names)} members", flush=True)
    if not a.no_control:
        oof, _ = nested(X, XT, y, a.C, a.nest, a.nest_seed, a.jobs)
        print(f"control       pooled {roc_auc_score(y, oof):.6f}", flush=True)
    coef = LogisticRegression(C=a.C, tol=1e-6, max_iter=20000).fit(X, y).coef_.ravel()
    order = np.argsort(-np.abs(coef))
    print("top members:", [(names[i], round(float(coef[i]), 3)) for i in order[:10]], flush=True)
    for k in a.top:
        idx = order[:k]
        H, HT = [], []
        for i in idx:
            for q in np.quantile(X[:, i], a.knots):
                H.append(np.maximum(X[:, i] - q, 0))
                HT.append(np.maximum(XT[:, i] - q, 0))
        H, HT = np.column_stack(H), np.column_stack(HT)
        hm, hs = H.mean(0), H.std(0) + 1e-9
        Xa = np.hstack([X, (H - hm) / hs])
        XTa = np.hstack([XT, (HT - hm) / hs])
        del H, HT
        oof, test = nested(Xa, XTa, y, a.C, a.nest, a.nest_seed, a.jobs)
        print(f"top {k:3d} x {len(a.knots)} knots pooled {roc_auc_score(y, oof):.6f}", flush=True)
        if a.write:
            np.save(RESULTS_DIR / f"oof_{a.write}_k{k}.npy", oof)
            np.save(RESULTS_DIR / f"test_{a.write}_k{k}.npy", test)
        del Xa, XTa


if __name__ == "__main__":
    main()
