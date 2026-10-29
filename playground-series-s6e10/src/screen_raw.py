"""Full-row incremental screen of raw public OOF/test CSV pairs against a finished stack (10-29).

    python src/screen_raw.py --stack pr42_lr_allown_c01_n10 OOF.csv:TEST.csv ...

Each CSV has (id, prediction); rows are aligned by id (train is id-sorted by build_features).
Prints solo AUC, corr with the stack, test-alignment rho, and the 5-fold LR delta of
(stack, candidate) over the stack alone; noise control first (null ~ -0.000001).
"""
import argparse
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src")
from features import TARGET, build_features

ap = argparse.ArgumentParser()
ap.add_argument("--stack", default="pr42_lr_allown_c01_n10")
ap.add_argument("pairs", nargs="+")
A = ap.parse_args()
tr = build_features(pl.read_csv("data/train.csv"))
y = tr[TARGET].to_numpy()
ids = tr["id"].to_numpy()
s = np.load(f"results/oof_{A.stack}.npy")
st = np.load(f"results/test_{A.stack}.npy")


def lg(p):
    p = np.asarray(p, float).ravel()
    if p.min() < 0 or p.max() > 1:
        return p
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def cv(X):
    o = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        o[b] = LogisticRegression(max_iter=500).fit(X[a], y[a]).decision_function(X[b])
    return roc_auc_score(y, o)


base = cv(s[:, None])
print("base", round(base, 6), flush=True)
rng = np.random.default_rng(0)
v = rng.normal(size=len(y))
print(f"{'noise':50s} {cv(np.c_[s, v]) - base:+.6f}", flush=True)
for pair in A.pairs:
    fo, ft = pair.split(":")
    o = pd.read_csv(fo).set_index("id").iloc[:, -1].reindex(ids).to_numpy()
    t = pd.read_csv(ft).sort_values("id").iloc[:, -1].to_numpy()
    if np.isnan(o).any() or len(t) != len(st):
        print(f"{fo}: misaligned / nan ({np.isnan(o).sum()} nan, test {len(t)})")
        continue
    lo, lt = lg(o), lg(t)
    z = (lo - lo.mean()) / lo.std()
    rho_o = spearmanr(lo[::50], s[::50]).statistic
    rho_t = spearmanr(lt[::50], st[::50]).statistic
    d = cv(np.c_[s, z]) - base
    print(f"{fo.split('/')[-2][:22]}/{fo.split('/')[-1]:26s} solo {roc_auc_score(y, lo):.6f} "
          f"rho oof {rho_o:.4f} test {rho_t:.4f}  delta {d:+.6f}", flush=True)
