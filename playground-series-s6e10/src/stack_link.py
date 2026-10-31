"""PR-58: the pr42 stacker (std, nest 10, L2) under other link functions — probit, cloglog, loglog.

    python src/stack_link.py --ours $(cat results/ours_all80_*.txt) --runs logit:0.01 probit:0.004 cloglog:0.01

The link changes how the likelihood weights rows (tails vs middle), hence the fitted member weights;
AUC only sees the linear score. Same L-BFGS fit as `stack_noise.py`, ε 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def link(z: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, dp/dz)."""
    if kind == "logit":
        p = expit(z)
        return p, p * (1 - p)
    if kind == "probit":
        return ndtr(z), np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi))
    if kind == "cloglog":
        e = np.exp(np.clip(z, -30, 5))
        return -np.expm1(-e), np.exp(np.clip(z, -30, 5) - e)
    if kind == "loglog":
        e = np.exp(np.clip(-z, -30, 5))
        return np.exp(-e), np.exp(np.clip(-z, -30, 5) - e)
    raise ValueError(kind)


def fit(X: np.ndarray, y: np.ndarray, C: float, kind: str, w0: np.ndarray | None) -> np.ndarray:
    d = X.shape[1]

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = (X @ w[:d].astype(np.float32) + w[d]).astype(np.float64)
        p, dp = link(z, kind)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        if kind == "probit":
            nll = -(y * log_ndtr(z) + (1 - y) * log_ndtr(-z)).sum()
        else:
            nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        g = (p - y) / (p * (1 - p)) * dp
        grad = np.empty(d + 1)
        grad[:d] = C * (X.T @ g.astype(np.float32)) + w[:d]
        grad[d] = C * g.sum()
        return C * nll + 0.5 * float(w[:d] @ w[:d]), grad

    x0 = np.zeros(d + 1) if w0 is None else w0
    return minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-6}).x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--nest", type=int, default=10)
    p.add_argument("--nest-seed", type=int, default=42)
    p.add_argument("--runs", nargs="+", default=["logit:0.01", "probit:0.004", "cloglog:0.01", "loglog:0.01"])
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(np.float64)
    names, O, _ = load_pairs(a.ours, [a.public])
    O = np.clip(O, -30, 30)
    X = ((O - O.mean(0)) / O.std(0)).astype(np.float32)
    del O
    print(f"{len(names)} members", flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=a.nest_seed).split(X, y))
    for run in a.runs:
        kind, C = run.split(":")
        oof = np.zeros(len(y))
        for tr, va in folds:
            w = fit(X[tr], y[tr], float(C), kind, None)
            oof[va] = X[va] @ w[:-1] + w[-1]
        print(f"{kind:8s} C {C:6s} {roc_auc_score(y, oof):.6f}", flush=True)


if __name__ == "__main__":
    main()
