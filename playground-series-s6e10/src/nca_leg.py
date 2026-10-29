"""PR-46: ModernNCA leg (Ye et al. 2024) — retrieval NN on the exported fold-safe matrices.

    python src/nca_leg.py --folds 1 --epochs 12          # fold-1 probe (saves val + test)

A PLR-embedded MLP maps each row to a d-dim space; the prediction for a query is the
softmax(-distance)-weighted label mean over training rows (all of them at inference). Training
uses stochastic neighbourhood sampling: each step's candidate set is a random subset of the
training rows, the batch's own rows masked out. Label-aware retrieval over the FULL 560k rows —
TabR (the only other member of this mechanism) was screened at 50k rows only.

Honest: fixed epoch count, last epoch used; the 5% inner holdout of the training rows is for
logging only and the scored validation fold never selects anything.
Per-fold outputs: results/_pt_nca<tag>/f{k}_{val,test}.npy (same layout as pytabkit_leg.py).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
RESULTS = Path(__file__).resolve().parents[1] / "results"


class PLR(nn.Module):
    """Periodic(-linear-ReLU) numeric embedding, one per feature."""

    def __init__(self, n: int, k: int = 16, d: int = 8, sigma: float = 0.1) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.w = nn.Parameter(torch.randn(n, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2 * math.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.cos(v), torch.sin(v)], -1)
        return F.relu(torch.einsum("bnk,nkd->bnd", v, self.w) + self.b).flatten(1)


class Encoder(nn.Module):
    def __init__(self, n_num: int, n_cat: int, d: int, hidden: int, drop: float) -> None:
        super().__init__()
        self.plr = PLR(n_num)
        self.net = nn.Sequential(
            nn.Linear(n_num * 8 + n_cat, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, d))

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.plr(xn), xc], 1))


def nca_logp(q: torch.Tensor, c: torch.Tensor, yc: torch.Tensor, temp: float,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """log p(y=1 | q) from softmax(-||q - c|| / temp) over candidates (binary)."""
    d = torch.cdist(q, c) / temp
    if mask is not None:
        d = d.masked_fill(mask, float("inf"))
    lse_all = torch.logsumexp(-d, 1)
    pos = (-d).masked_fill(~yc.bool()[None], float("-inf"))
    neg = (-d).masked_fill(yc.bool()[None], float("-inf"))
    return torch.stack([torch.logsumexp(neg, 1), torch.logsumexp(pos, 1)], 1) - lse_all[:, None]


@torch.no_grad()
def embed(enc: Encoder, xn: torch.Tensor, xc: torch.Tensor, bs: int = 16384) -> torch.Tensor:
    enc.eval()
    return torch.cat([enc(xn[i:i + bs], xc[i:i + bs]) for i in range(0, len(xn), bs)])


@torch.no_grad()
def predict(enc: Encoder, cand: torch.Tensor, yc: torch.Tensor, xn: torch.Tensor, xc: torch.Tensor,
            temp: float, qbs: int = 512) -> np.ndarray:
    """Logit of p(y=1) for every query, all training rows as candidates."""
    q = embed(enc, xn, xc)
    out = []
    for i in range(0, len(q), qbs):
        lp = nca_logp(q[i:i + qbs], cand, yc, temp)
        out.append((lp[:, 1] - lp[:, 0]).float().cpu())
    return torch.cat(out).numpy()


def prep(z: dict, cat_idx: list[int]) -> tuple[dict, int, int]:
    num_idx = [i for i in range(z["x_tr"].shape[1]) if i not in cat_idx]
    med = np.nanmedian(z["x_tr"], axis=0)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                             random_state=0)
    out = {}
    for key in ("x_tr", "x_va", "x_te"):
        x = z[key].astype(np.float32)
        r, c = np.where(np.isnan(x))
        x[r, c] = med[c]
        z[key] = x
    qt.fit(z["x_tr"][:, num_idx])
    levels = [np.unique(z["x_tr"][:, i]) for i in cat_idx]
    for key in ("x_tr", "x_va", "x_te"):
        xn = qt.transform(z[key][:, num_idx]).astype(np.float32)
        xc = np.concatenate([(z[key][:, [i]] == lv[None]).astype(np.float32)
                             for i, lv in zip(cat_idx, levels)], 1)
        out[key] = (xn, xc)
    return out, len(num_idx), out["x_tr"][1].shape[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[1])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--drop", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--cand", type=int, default=40_000, help="candidates sampled per step")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--no-test", action="store_true")
    a = p.parse_args()
    out = RESULTS / f"_pt_nca{a.tag}"
    out.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    for k in a.folds:
        if (out / f"f{k}_val.npy").exists():
            print(f"fold {k}: done already")
            continue
        torch.manual_seed(a.seed + k)
        rng = np.random.default_rng(a.seed + k)
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        mats, n_num, n_cat = prep(z, [int(i) for i in z["cat_idx"]])
        ytr = z["y_tr"].astype(np.int64)
        n = len(ytr)
        hold = rng.choice(n, n // 20, replace=False)  # inner logging holdout only
        fit = np.setdiff1d(np.arange(n), hold)
        Xn = torch.tensor(mats["x_tr"][0], device=dev)
        Xc = torch.tensor(mats["x_tr"][1], device=dev)
        Y = torch.tensor(ytr, device=dev)
        enc = Encoder(n_num, n_cat, a.d, a.hidden, a.drop).to(dev)
        opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * (len(fit) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        fit_t = torch.tensor(fit, device=dev)
        t0 = time.time()
        for ep in range(a.epochs):
            enc.train()
            perm = fit_t[torch.randperm(len(fit_t), device=dev)]
            tot = 0.0
            for s in range(len(fit) // a.bs):
                qi = perm[s * a.bs:(s + 1) * a.bs]
                ci = fit_t[torch.randint(len(fit_t), (a.cand,), device=dev)]
                both = torch.cat([qi, ci])
                e = enc(Xn[both], Xc[both])
                mask = qi[:, None] == ci[None]
                lp = nca_logp(e[:a.bs], e[a.bs:], Y[ci], a.temp, mask)
                loss = F.nll_loss(lp, Y[qi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            # inner holdout (training rows only), candidates = fit rows
            cand = embed(enc, Xn[fit_t], Xc[fit_t])
            ph = predict(enc, cand, Y[fit_t], Xn[hold], Xc[hold], a.temp)
            print(f"fold {k} ep {ep}: loss {tot / (len(fit) // a.bs):.5f} inner AUC "
                  f"{roc_auc_score(ytr[hold], ph):.5f} ({time.time() - t0:.0f}s)", flush=True)
            del cand
        # last epoch, all training rows as candidates
        cand = embed(enc, Xn, Xc)
        va = [torch.tensor(m, device=dev) for m in mats["x_va"]]
        pv = predict(enc, cand, Y, *va, a.temp)
        print(f"fold {k}: AUC {roc_auc_score(z['y_va'], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), pv]))
        if not a.no_test:
            te = [torch.tensor(m, device=dev) for m in mats["x_te"]]
            np.save(out / f"f{k}_test.npy", predict(enc, cand, Y, *te, a.temp))
            del te
        del enc, opt, cand, va, Xn, Xc, Y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
