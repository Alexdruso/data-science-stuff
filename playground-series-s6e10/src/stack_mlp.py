"""MLP stacker over member logits (nested on the base folds), residual to the LR stack: the net's
output is added to a frozen linear term initialised from the fold's LR, so it starts at LR.

    python src/stack_mlp.py --ours LEG ...
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
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--wd", type=float, default=1e-3)
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    _, O, _ = load_pairs(a.ours, [str(DATA_DIR / "public_oof" / "_pool")])
    oof_lr, oof = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(O, y)):
        sc = StandardScaler().fit(O[tr])
        xtr, xva = sc.transform(O[tr]).astype(np.float32), sc.transform(O[va]).astype(np.float32)
        lr = LogisticRegression(C=0.1, max_iter=3000).fit(xtr, y[tr])
        oof_lr[va] = lr.decision_function(xva)
        torch.manual_seed(k)
        lin = nn.Linear(xtr.shape[1], 1).cuda()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))
        for q in lin.parameters():
            q.requires_grad_(False)
        net = nn.Sequential(nn.Dropout(0.1), nn.Linear(xtr.shape[1], a.hidden), nn.GELU(),
                            nn.Dropout(0.2), nn.Linear(a.hidden, 1)).cuda()
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=a.wd)
        X, Y = torch.from_numpy(xtr).cuda(), torch.from_numpy(y[tr].astype(np.float32)).cuda()
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(a.epochs):
            net.train()
            perm = torch.randperm(len(X), device="cuda")
            for i in range(0, len(X), 4096):
                b = perm[i:i + 4096]
                loss = lossf((lin(X[b]) + net(X[b])).squeeze(1), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            V = torch.from_numpy(xva).cuda()
            oof[va] = (lin(V) + net(V)).squeeze(1).cpu().numpy()
        print(f"fold {k + 1}: LR {roc_auc_score(y[va], oof_lr[va]):.6f}  +MLP {roc_auc_score(y[va], oof[va]):.6f}",
              flush=True)
        del X, Y, V, net, lin, opt
        torch.cuda.empty_cache()
    print(f"LR {roc_auc_score(y, oof_lr):.6f}   LR+MLP residual {roc_auc_score(y, oof):.6f}")


if __name__ == "__main__":
    main()
