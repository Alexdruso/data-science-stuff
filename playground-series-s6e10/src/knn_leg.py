"""PR-27: GPU kNN leg on the exported TFM fold matrices (route TE + og + agg + aux).

    python src/knn_leg.py --fold 1 --n-val 20000        # probe
    python src/knn_leg.py --all --k 100                  # 5 folds + test (test by fold-1 index)

Columns are standardised on the fold's training rows, then scaled by |Spearman| with the
label on those rows (a cheap supervised metric). Prediction = smoothed, distance-weighted
mean label of the k nearest training rows. Brute force on the GPU in chunks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).resolve().parents[1] / "results"


def knn_predict(xtr: torch.Tensor, ytr: torch.Tensor, xq: torch.Tensor, ks: list[int]) -> dict[int, np.ndarray]:
    out = {k: [] for k in ks}
    kmax = max(ks)
    tr_sq = (xtr * xtr).sum(1)
    prior = ytr.mean()
    for s in range(0, len(xq), 256):
        q = xq[s : s + 256]
        d = tr_sq[None, :] - 2 * q @ xtr.T  # + |q|^2 is constant per row
        dist, idx = torch.topk(d, kmax, dim=1, largest=False)
        lab = ytr[idx]
        for k in ks:
            w = torch.ones_like(dist[:, :k])
            out[k].append(((lab[:, :k] * w).sum(1) + 5 * prior) / (w.sum(1) + 5))
        del d, dist, idx, lab
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--all", action="store_true")
    p.add_argument("--k", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--tag", default="knn")
    a = p.parse_args()
    dev = "cuda"
    folds = range(1, 6) if a.all else [a.fold]
    oof: dict[int, np.ndarray] = {}
    test: dict[int, np.ndarray] = {}
    for f in folds:
        t0 = time.time()
        z = dict(np.load(RESULTS / f"tfm_fold{f}.npz", allow_pickle=True))
        xtr, ytr = np.nan_to_num(z["x_tr"].astype(np.float32)), z["y_tr"].astype(np.float32)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        sub = np.random.default_rng(f).choice(len(xtr), 100_000, replace=False)
        w = np.array([abs(spearmanr(xtr[sub, j], ytr[sub]).statistic) for j in range(xtr.shape[1])])
        w = np.nan_to_num(w).astype(np.float32)

        def prep(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor((np.nan_to_num(x.astype(np.float32)) - mu) / sd * w, device=dev)

        Xtr, Ytr = prep(xtr), torch.as_tensor(ytr, device=dev)
        xv = z["x_va"] if a.n_val is None else z["x_va"][: a.n_val]
        yv = z["y_va"] if a.n_val is None else z["y_va"][: a.n_val]
        pv = knn_predict(Xtr, Ytr, prep(xv), a.k)
        print(f"fold {f} ({time.time() - t0:.0f}s): " + "  ".join(f"k{k} {roc_auc_score(yv, pv[k]):.5f}" for k in a.k), flush=True)
        if a.all:
            for k in a.k:
                oof.setdefault(k, np.zeros(len(z["y_tr"]) + len(z["y_va"])))[z["va_idx"]] = pv[k]
            if f == 1:
                pt = knn_predict(Xtr, Ytr, prep(z["x_te"]), a.k)
                test = pt
        del Xtr, Ytr
        torch.cuda.empty_cache()
    if a.all:
        for k in a.k:
            np.save(RESULTS / f"oof_{a.tag}_k{k}.npy", oof[k])
            np.save(RESULTS / f"test_{a.tag}_k{k}.npy", test[k])
        print("saved")


if __name__ == "__main__":
    main()
