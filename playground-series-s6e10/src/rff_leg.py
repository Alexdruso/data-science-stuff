"""PR-54: random-Fourier-feature kernel logistic regression (RBF kernel machine) member.

    python src/rff_leg.py --folds 1 --dim 8192 --gamma 0.05

On the 69 exported fold-safe columns (tfm_fold{k}.npz): training-fold medians fill NaN,
quantile-normal transform fit on the training rows, z = sqrt(2/D) cos(XW + b) with
W ~ N(0, 2·gamma), then an L2 logistic regression on z, full-data minibatch Adam on GPU for a
fixed number of epochs (no early stopping on any scored rows → honest). A smooth kernel machine:
no kind like it in the pool (GBDTs, NNs with learned features, TFMs, GAM, FMs, kNN).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

RESULTS = Path(__file__).parent.parent / "results"


def feats(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cos(x @ W + b) * (2.0 / W.shape[1]) ** 0.5


def predict(x: np.ndarray, W, b, w, c, dev) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32768):
            xb = torch.from_numpy(x[i:i + 32768]).to(dev)
            out.append((feats(xb, W, b) @ w + c).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--tag", default="rff_leg")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = RESULTS / a.tag
    out.mkdir(exist_ok=True)
    for k in a.folds:
        z = np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)
        xtr, xva, xte = (z[n].astype(np.float32) for n in ("x_tr", "x_va", "x_te"))
        med = np.nanmedian(xtr, axis=0)
        for x in (xtr, xva, xte):
            idx = np.where(np.isnan(x))
            x[idx] = med[idx[1]]
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                                 random_state=0).fit(xtr)
        xtr, xva, xte = (qt.transform(x).astype(np.float32) for x in (xtr, xva, xte))
        g = torch.Generator(device="cpu").manual_seed(k)
        W = (torch.randn(xtr.shape[1], a.dim, generator=g) * (2 * a.gamma) ** 0.5).to(dev)
        b = (torch.rand(a.dim, generator=g) * 2 * np.pi).to(dev)
        w = torch.zeros(a.dim, device=dev, requires_grad=True)
        c = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, c], lr=3e-3)
        ytr = torch.from_numpy(z["y_tr"].astype(np.float32))
        n, bs = len(xtr), 4096
        steps = a.epochs * (n // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
        t0 = time.time()
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(n // bs):
                j = perm[i * bs:(i + 1) * bs]
                xb = torch.from_numpy(xtr[j.numpy()]).to(dev)
                logit = feats(xb, W, b) @ w + c
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, ytr[j].to(dev))
                loss = loss + a.l2 * (w * w).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            pv = predict(xva, W, b, w.detach(), c.detach(), dev)
            print(f"fold {k} ep {ep} loss {loss.item():.4f} val AUC {roc_auc_score(z['y_va'], pv):.5f} "
                  f"({time.time() - t0:.0f}s) [log only, last epoch is used]", flush=True)
        pt = predict(xte, W, b, w.detach(), c.detach(), dev)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), pv]))
        np.save(out / f"f{k}_test.npy", pt)
        del W, b, w, c, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
