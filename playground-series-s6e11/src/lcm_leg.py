"""Class-conditional latent-class mixture (mixture of independent categoricals) scored by Bayes.

    python src/lcm_leg.py --fold 1 [--k0 64 --k1 16 --bins 32 --iters 40]
Writes results/_lcm<tag>/f{fold}_val.npy (LLR) for lm_pool_eval.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS, RAW_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def encode(tr: pl.DataFrame, fit: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    cols = []
    for c in RAW_COLS:
        v = tr[c].to_numpy()
        if c in NUM_COLS and len(np.unique(v[fit])) > bins:
            edges = np.unique(np.quantile(v[fit].astype(float), np.linspace(0, 1, bins + 1)[1:-1]))
            cols.append(np.searchsorted(edges, v.astype(float), side="right"))
        else:
            _, inv = np.unique(v.astype(str), return_inverse=True)
            cols.append(inv)
    X = np.column_stack(cols).astype(np.int64)
    return X, int(X.max()) + 1


def em(X: np.ndarray, K: int, B: int, iters: int, alpha: float, seed: int
       ) -> tuple[np.ndarray, np.ndarray]:
    n, D = X.shape
    rng = np.random.default_rng(seed)
    r = rng.dirichlet(np.ones(K), n)
    for _ in range(iters):
        nk = r.sum(0)
        logpi = np.log(nk / n)
        th = np.empty((K, D, B))
        for j in range(D):
            for k in range(K):
                th[k, j] = np.bincount(X[:, j], weights=r[:, k], minlength=B) + alpha
        th /= th.sum(2, keepdims=True)
        lth = np.log(th)
        L = np.tile(logpi, (n, 1))
        for j in range(D):
            L += lth[:, j, X[:, j]].T
        r = np.exp(L - logsumexp(L, 1, keepdims=True))
    return logpi, lth


def loglik(X: np.ndarray, logpi: np.ndarray, lth: np.ndarray) -> np.ndarray:
    L = np.tile(logpi, (len(X), 1))
    for j in range(X.shape[1]):
        L += lth[:, j, X[:, j]].T
    return logsumexp(L, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k0", type=int, default=64)
    ap.add_argument("--k1", type=int, default=16)
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--neg_frac", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = tr[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    k = a.fold
    trn = np.where(folds != k)[0]
    va = np.where(folds == k)[0]
    X, B = encode(tr, trn, a.bins)
    rng = np.random.default_rng(1000 + k)
    neg = trn[(y[trn] == 0) & (rng.random(len(trn)) < a.neg_frac)]
    pos = trn[y[trn] == 1]
    lp1, th1 = em(X[pos], a.k1, B, a.iters, a.alpha, 1)
    lp0, th0 = em(X[neg], a.k0, B, a.iters, a.alpha, 0)
    s = loglik(X[va], lp1, th1) - loglik(X[va], lp0, th0)
    out = RESULTS_DIR / f"_lcm{a.tag}"
    out.mkdir(exist_ok=True)
    np.save(out / f"f{k}_val.npy", s)
    print(f"fold {k}: lcm K0 {a.k0} K1 {a.k1} solo {roc_auc_score(y[va], s):.5f}", flush=True)


if __name__ == "__main__":
    main()
