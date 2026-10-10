"""PR-28: swap-noise denoising autoencoder (label-free, train + test) → MLP on its activations.

    python src/dae_leg.py --fold 1          # screen: fold-1 val predictions → probe file
    python src/dae_leg.py --all --tag dae   # 5 folds + test

The DAE sees every synthetic row (train + test), no labels; its three hidden layers
are the representation. The supervised head trains on the fold's training rows for a
fixed number of epochs (no epoch selection on the scored fold).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erfinv
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch import nn

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEV = "cuda"
RATINGS = 13


def rank_gauss(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(v, nan=np.nanmedian(v))
    r = (rankdata(v) - 0.5) / len(v)
    return (np.sqrt(2) * erfinv(2 * r - 1)).astype(np.float32)


def build_inputs() -> tuple[np.ndarray, np.ndarray, int]:
    """All synthetic rows (train in id order, then test) → DAE input matrix."""
    z = np.load(RESULTS / "tfm_fold1.npz", allow_pickle=True)
    cols = list(z["cols"])
    n_tr = len(z["y_tr"]) + len(z["y_va"])
    x = np.zeros((n_tr, len(cols)), np.float32)
    x[np.setdiff1d(np.arange(n_tr), z["va_idx"])] = z["x_tr"]
    x[z["va_idx"]] = z["x_va"]
    y = np.zeros(n_tr, np.float32)
    y[np.setdiff1d(np.arange(n_tr), z["va_idx"])] = z["y_tr"]
    y[z["va_idx"]] = z["y_va"]
    allx = np.vstack([x, z["x_te"]])
    blocks = []
    for c in cols[2:2 + RATINGS] + ["gender", "customer_type", "type_of_travel", "class"]:
        v = allx[:, cols.index(c)].astype(int)
        blocks.append((v[:, None] == np.unique(v)[None, :]).astype(np.float32))
    label_free = [c for c in cols if not c.startswith("TE_")
                  and c not in cols[2:2 + RATINGS] + ["gender", "customer_type", "type_of_travel", "class"]]
    blocks.append(np.column_stack([rank_gauss(allx[:, cols.index(c)]) for c in label_free]))
    return np.hstack(blocks), y, n_tr


class DAE(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(d, h), nn.Linear(h, h), nn.Linear(h, h)
        self.out = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = torch.relu(self.l1(x))
        a2 = torch.relu(self.l2(a1))
        a3 = torch.relu(self.l3(a2))
        return self.out(a3), torch.cat([a1, a2, a3], 1)


def train_dae(X: np.ndarray, h: int, epochs: int, noise: float, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X).to(DEV)
    n, d = Xt.shape
    m = DAE(d, h).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    bs = 1024
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(DEV)
        tot = 0.0
        for i in range(0, n, bs):
            b = Xt[perm[i:i + bs]]
            # swap noise: each cell replaced, with prob `noise`, by the same column of a random row
            mask = torch.rand_like(b) < noise
            donor = Xt[torch.randint(0, n, (len(b),), device=DEV)]
            rec, _ = m(torch.where(mask, donor, b))
            loss = ((rec - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"   dae epoch {ep + 1}: {tot / n:.5f}", flush=True)
    m.eval()
    feats = np.zeros((n, 3 * h), np.float16)
    with torch.no_grad():
        for i in range(0, n, 8192):
            feats[i:i + 8192] = m(Xt[i:i + 8192])[1].half().cpu().numpy()
    del m, opt, Xt
    torch.cuda.empty_cache()
    return feats


def train_head(F: np.ndarray, y: np.ndarray, tr: np.ndarray, pred: list[np.ndarray],
               epochs: int, seed: int) -> list[np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = F.shape[1]
    m = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
                      nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / 2048))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    yt = torch.from_numpy(y).to(DEV)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            xb = torch.from_numpy(F[b].astype(np.float32)).to(DEV)
            loss = lossf(m(xb).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for idx in pred:
            outs.append(np.concatenate([m(torch.from_numpy(F[idx[i:i + 16384]].astype(np.float32)).to(DEV))
                                        .squeeze(1).cpu().numpy() for i in range(0, len(idx), 16384)]))
    del m, opt, yt
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--all", action="store_true")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dae-epochs", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--head-epochs", type=int, default=4)
    p.add_argument("--tag", default="dae")
    a = p.parse_args()

    t0 = time.time()
    X, y, n_tr = build_inputs()
    print(f"inputs {X.shape} ({time.time() - t0:.0f}s)", flush=True)
    F = train_dae(X, a.hidden, a.dae_epochs, a.noise, 0)
    del X
    print(f"dae done ({time.time() - t0:.0f}s)", flush=True)
    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_tr), y[:n_tr]))
    folds = range(1, 6) if a.all else [a.fold]
    oof, test = np.zeros(n_tr), np.zeros(len(F) - n_tr)
    te_idx = np.arange(n_tr, len(F))
    for k in folds:
        tr, va = splits[k - 1]
        pv, pt = train_head(F, y, tr, [va, te_idx], a.head_epochs, k)
        oof[va] = pv
        test += pt / len(folds)
        print(f"fold {k}: AUC {roc_auc_score(y[va], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        if not a.all:
            np.save(RESULTS / f"probe_{a.tag}_f{k}.npy", np.stack([va.astype(float), pv]))
        torch.cuda.empty_cache()
    if a.all:
        print(f"OOF AUC {roc_auc_score(y[:n_tr], oof):.6f}")
        np.save(RESULTS / f"oof_{a.tag}.npy", oof)
        np.save(RESULTS / f"test_{a.tag}.npy", test)


if __name__ == "__main__":
    main()
