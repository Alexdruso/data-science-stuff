"""AUC-direct stacker: per outer fold, LR on standardised member logits, then the weights
are fine-tuned on a pairwise soft-AUC surrogate (softplus of negative pos−neg margin / tau)
with an L2 pull back to the LR solution. Nested on the base folds, scored like stack_public.

    python src/stack_auc.py --ours LEG ... [--lam 1e-3] [--tau 0.5] [--write NAME]
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def finetune(X: np.ndarray, y: np.ndarray, w0: np.ndarray, lam: float, tau: float, steps: int,
             seed: int) -> np.ndarray:
    g = torch.Generator(device="cuda").manual_seed(seed)
    Xt = torch.from_numpy(X.astype(np.float32)).cuda()
    pos = torch.from_numpy(np.where(y == 1)[0]).cuda()
    neg = torch.from_numpy(np.where(y == 0)[0]).cuda()
    w0t = torch.from_numpy(w0.astype(np.float32)).cuda()
    w = w0t.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=2e-3)
    for _ in range(steps):
        i = pos[torch.randint(len(pos), (200_000,), device="cuda", generator=g)]
        j = neg[torch.randint(len(neg), (200_000,), device="cuda", generator=g)]
        margin = (Xt[i] - Xt[j]) @ w
        loss = torch.nn.functional.softplus(-margin / tau).mean() + lam * ((w - w0t) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = w.detach().cpu().numpy().astype(np.float64)
    del Xt, pos, neg, w0t, w, opt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--lam", type=float, nargs="+", default=[1e-3])
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    _, O, T = load_pairs(a.ours, [str(DATA_DIR / "public_oof" / "_pool")])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(O, y))
    base = {}
    for k, (tr, va) in enumerate(folds):
        sc = StandardScaler().fit(O[tr])
        lr = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(O[tr]), y[tr])
        base[k] = (sc, lr.coef_.ravel())
    oof_lr = np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        oof_lr[va] = base[k][0].transform(O[va]) @ base[k][1]
    print(f"LR (no intercept) {roc_auc_score(y, oof_lr):.6f}", flush=True)
    for lam in a.lam:
        oof, test = np.zeros(len(y)), np.zeros(len(T))
        for k, (tr, va) in enumerate(folds):
            sc, w0 = base[k]
            w = finetune(sc.transform(O[tr]), y[tr], w0, lam, a.tau, a.steps, k)
            oof[va] = sc.transform(O[va]) @ w
            test += sc.transform(T) @ w / 5
        print(f"AUC-finetune lam={lam} tau={a.tau}: {roc_auc_score(y, oof):.6f}", flush=True)
        if a.write:
            np.save(RESULTS_DIR / f"oof_{a.write}_lam{lam}.npy", oof)
            np.save(RESULTS_DIR / f"test_{a.write}_lam{lam}.npy", test)


if __name__ == "__main__":
    main()
