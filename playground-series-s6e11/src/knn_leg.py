"""kNN leg: neighbour fraud rate (prior-corrected) against fold-train reference = all positives + R negatives.
    python src/knn_leg.py --k 100 --neg 120000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import BOOL_COLS, CAT_COLS, NUM_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=100)
ap.add_argument("--neg", type=int, default=120000); a = ap.parse_args()
tr = build_features(pl.read_csv(DATA_DIR / "train.csv")); te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
y = tr[TARGET].to_numpy().astype(int); folds = fold_assignment(y)
al = pl.concat([tr.drop(TARGET), te], how="vertical_relaxed")
cols = []
for c in NUM_COLS:
    v = al[c].to_numpy().astype(float); cols.append(ndtri(rankdata(v) / (len(v) + 1)))
for c in CAT_COLS:
    v = al[c].to_numpy().astype(str)
    for u in np.unique(v):
        cols.append((v == u) * 1.0)
for c in BOOL_COLS:
    v = al[c].cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "1.0"]).to_numpy() * 1.0
    cols.append((v - v.mean()) / v.std())
X = np.column_stack(cols).astype(np.float32); Xtr, Xte = X[: len(y)], X[len(y):]
dev = "cpu"
torch.set_num_threads(5)


def knn_score(ref, ry, rw, q):
    R = torch.from_numpy(ref); r2 = (R * R).sum(1)
    out = np.zeros(len(q))
    for s in range(0, len(q), 4096):
        Q = torch.from_numpy(q[s:s + 4096])
        d = r2[None] - 2 * Q @ R.T
        idx = torch.topk(d, a.k, largest=False).indices.numpy()
        w = rw[idx]; out[s:s + 4096] = (w * ry[idx]).sum(1) / w.sum(1)
    return out


oof, tst = np.zeros(len(y)), np.zeros(len(Xte)); rng = np.random.default_rng(0)
for k in range(1, 6):
    fit = np.where(folds != k)[0]; pos = fit[y[fit] == 1]; neg = fit[y[fit] == 0]
    negs = rng.choice(neg, a.neg, replace=False); ref = np.concatenate([pos, negs])
    rw = np.where(y[ref] == 1, 1.0, len(neg) / a.neg)
    va = np.where(folds == k)[0]
    oof[va] = knn_score(Xtr[ref], y[ref], rw, Xtr[va])
    tst += knn_score(Xtr[ref], y[ref], rw, Xte) / 5
    print(k, roc_auc_score(y[va], oof[va]), flush=True)
auc = roc_auc_score(y, oof); n = f"knn_k{a.k}"; print(n, auc)
save_predictions(RESULTS_DIR, n, oof=oof, test=tst, train_ids=tr["id"].to_numpy(), test_ids=te["id"].to_numpy())
save_cv_result(RESULTS_DIR, n, [], auc, metric_name="auc", params=vars(a))
