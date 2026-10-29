"""PR-47: label-noise-corrected LR stacker — the pr42 recipe (standardised members, C 0.01, nest 10)
fit with the forward-corrected likelihood P(y=1|z) = eps + (1 - 2 eps) sigmoid(z) (symmetric flip
rate eps). Same L2 objective as sklearn (C * sum(nll) + 0.5 ||w||^2, intercept unpenalised), solved
with L-BFGS; eps = 0 is the control and must reproduce pr42 (0.962237). Ranking uses z.

    python src/stack_noise.py --ours $(cat results/ours_all80_*.txt) --eps 0 0.01 0.02 0.04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def fit(X: np.ndarray, y: np.ndarray, C: float, eps: float, w0: np.ndarray | None) -> np.ndarray:
    d = X.shape[1]

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w[:d].astype(np.float32) + w[d]
        s = expit(z.astype(np.float64))
        p = np.clip(eps + (1 - 2 * eps) * s, 1e-12, 1 - 1e-12)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        g = (p - y) / (p * (1 - p)) * (1 - 2 * eps) * s * (1 - s)
        grad = np.empty(d + 1)
        grad[:d] = C * (X.T @ g.astype(np.float32)) + w[:d]
        grad[d] = C * g.sum()
        return C * nll + 0.5 * float(w[:d] @ w[:d]), grad

    x0 = np.zeros(d + 1) if w0 is None else w0
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-6})
    return r.x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--nest", type=int, default=10)
    p.add_argument("--nest-seed", type=int, default=42)
    p.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04])
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(np.float64)
    names, O, T = load_pairs(a.ours, [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    mu, sd = O.mean(0), O.std(0)
    X, XT = ((O - mu) / sd).astype(np.float32), ((T - mu) / sd).astype(np.float32)
    del O, T
    print(f"{len(names)} members", flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=a.nest_seed).split(X, y))
    warm: list[np.ndarray | None] = [None] * a.nest
    for eps in a.eps:
        oof, test = np.zeros(len(y)), np.zeros(len(XT))
        for j, (tr, va) in enumerate(folds):
            # C scales with the training rows exactly as sklearn's per-fold fit does
            w = fit(X[tr], y[tr], a.C, eps, warm[j])
            warm[j] = w
            oof[va] = X[va] @ w[:-1] + w[-1]
            test += (XT @ w[:-1] + w[-1]) / a.nest
        print(f"eps {eps:.3f} pooled {roc_auc_score(y, oof):.6f}", flush=True)
        if a.write:
            np.save(RESULTS_DIR / f"oof_{a.write}_e{eps:g}.npy", oof)
            np.save(RESULTS_DIR / f"test_{a.write}_e{eps:g}.npy", test)


if __name__ == "__main__":
    main()
