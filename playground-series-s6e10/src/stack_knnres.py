"""PR-56: kNN residual corrector in member-logit space (exploratory screen, 10-31).

    python src/stack_knnres.py --ours $(cat results/ours_all80_*.txt) --k 50 200 --pcs 16 64

Members are standardised (pr42 geometry) and projected on their top PCs (fit on train). For each
row, the mean pr42 residual y - sigmoid(s) of its k nearest rows WITHIN ITS OWN BASE FOLD (self
excluded): every member that scored those rows was trained on the other four folds, so no
neighbour's residual carries the row's label (the route-TE artifact of 10-11 needs a cross-fold
neighbour). Read = 5-fold LR on (s, knn_res) minus LR on (s, noise). GPU for the distances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def knn_mean(Z: np.ndarray, r: np.ndarray, ks: list[int], dev: torch.device) -> dict[int, np.ndarray]:
    z = torch.from_numpy(Z).to(dev)
    rr = torch.from_numpy(r.astype(np.float32)).to(dev)
    sq = (z * z).sum(1)
    out = {k: np.zeros(len(Z), np.float32) for k in ks}
    kmax = max(ks)
    for i in range(0, len(Z), 2048):
        q = z[i:i + 2048]
        d = (q * q).sum(1, keepdim=True) + sq[None] - 2 * q @ z.T
        d[torch.arange(len(q)), torch.arange(i, i + len(q))] = float("inf")
        idx = d.topk(kmax, largest=False).indices
        nr = rr[idx]
        for k in ks:
            out[k][i:i + len(q)] = nr[:, :k].mean(1).cpu().numpy()
        del d, idx, nr
    del z, rr, sq
    torch.cuda.empty_cache()
    return out


def read(s: np.ndarray, f: np.ndarray, y: np.ndarray) -> float:
    o1, o2 = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(s, y):
        X = np.column_stack([s, f])
        o2[va] = LogisticRegression().fit(X[tr], y[tr]).decision_function(X[va])
        o1[va] = s[va]
    return roc_auc_score(y, o2) - roc_auc_score(y, o1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--stack", default="pr42_lr_allown_c01_n10")
    p.add_argument("--k", type=int, nargs="+", default=[50, 200])
    p.add_argument("--pcs", type=int, nargs="+", default=[16, 64])
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    s = np.load(RESULTS_DIR / f"oof_{a.stack}.npy")
    r = y - 1 / (1 + np.exp(-s))
    _, O, _ = load_pairs(a.ours, [a.public])
    O = np.clip(O, -30, 30)
    X = ((O - O.mean(0)) / O.std(0)).astype(np.float32)
    del O
    _, _, Vt = np.linalg.svd(X[::20] - X[::20].mean(0), full_matrices=False)
    rng = np.random.default_rng(0)
    print(f"noise {read(s, rng.normal(size=len(y)), y):+.6f}", flush=True)
    dev = torch.device("cuda")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    for npc in a.pcs:
        Z = X @ Vt[:npc].T
        Z = (Z / Z.std(0)).astype(np.float32)
        feats = {k: np.zeros(len(y)) for k in a.k}
        for _, va in folds:
            m = knn_mean(Z[va], r[va], a.k, dev)
            for k in a.k:
                feats[k][va] = m[k]
        for k in a.k:
            print(f"pcs {npc:3d} k {k:4d}  corr(f, r) {np.corrcoef(feats[k], r)[0, 1]:+.4f}  "
                  f"delta {read(s, feats[k], y):+.6f}", flush=True)


if __name__ == "__main__":
    main()
