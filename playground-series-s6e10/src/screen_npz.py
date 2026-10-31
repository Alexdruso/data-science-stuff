"""Full-row screen of a public npz of `<name>__oof` / `<name>__test` arrays (id order) vs a finished stack (10-31).

    python src/screen_npz.py PATH.npz [--stack pr42_lr_allown_c01_n10]

Per member: solo AUC, max |corr| with the current `_pool` logits (near-dup), and the 5-fold LR delta of
(stack, member) over (stack, noise). Same read as `screen_raw.py`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src")
from features import TARGET, build_features

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--stack", default="pr42_lr_allown_c01_n10")
A = ap.parse_args()
y = build_features(pl.read_csv("data/train.csv"))[TARGET].to_numpy()
s = np.load(f"results/oof_{A.stack}.npy")
folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(s, y))


def lg(p):
    p = p.astype(np.float64)
    if p.min() < 0 or p.max() > 1:
        return p
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def read(f):
    o = np.zeros(len(y))
    for tr, va in folds:
        X = np.column_stack([s, f])
        o[va] = LogisticRegression().fit(X[tr], y[tr]).decision_function(X[va])
    return roc_auc_score(y, o)


pool = np.column_stack([lg(np.load(f)) for f in sorted(Path("data/public_oof/_pool").glob("oof_*.npy"))])[::7]
pool = (pool - pool.mean(0)) / pool.std(0)
base = read(np.random.default_rng(0).normal(size=len(y)))
print(f"noise-base {base:.6f}", flush=True)
z = np.load(A.npz)
names = sorted({k.rsplit("__", 1)[0] for k in z.files if k.endswith("__oof")})
for n in names:
    if n + "__test" not in z.files:
        continue
    o = lg(z[n + "__oof"])
    if np.isnan(o).any() or len(o) != len(y):
        print(f"{n:55s} bad", flush=True)
        continue
    oz = (o[::7] - o[::7].mean()) / o[::7].std()
    mc = np.abs(pool.T @ oz / len(oz)).max()
    print(f"{n:55s} solo {roc_auc_score(y, o):.6f} maxcorr {mc:.5f} delta {read(o) - base:+.6f}", flush=True)
