"""Factorization-machine leg over high-cardinality value tokens (CTR-style).

Every column contributes several tokens taken from its CSV text: the full value, and for the
floats the int part, the first decimal, the int part + first decimal, the leading two digits and
the number of decimals. Tokens seen < MIN_CNT times in train+test map to a per-field "rare"
token (label-free, so fold-safe). logit = b + sum_i w_i + sum_{i<j} <v_i, v_j> (rank k).

    ~/venvs/lm/bin/python src/fm_leg.py --folds 1 [--k 16] [--epochs 4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS, RAW_COLS, TARGET

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]


def tokens(df: pd.DataFrame, min_cnt: int, drop: tuple = ()) -> tuple[np.ndarray, int]:
    fields = {}
    for c in RAW_COLS:
        s = df[c].astype(str)
        if not (c in FLOATS and "full" in drop):
            fields[c] = s
        if c in FLOATS:
            ip = s.str.split(".").str[0]
            fr = s.str.split(".").str[1].fillna("")
            fields[c + "_ip"] = ip
            fields[c + "_d1"] = fr.str[:1]
            if "ipd1" not in drop:
                fields[c + "_ipd1"] = ip + "." + fr.str[:1]
            fields[c + "_lead2"] = ip.str[:2] + "_" + ip.str.len().astype(str)
            fields[c + "_nd"] = fr.str.len().astype(str)
            if "last2" not in drop:
                fields[c + "_last2"] = fr.str[-2:]
    cols, off = [], 0
    for name, s in fields.items():
        vc = s.value_counts()
        keep = vc[vc >= min_cnt].index
        codes = pd.Categorical(s.where(s.isin(keep), "__rare__")).codes.astype(np.int64)
        cols.append(codes + off)
        off += int(codes.max()) + 1
    print(f"  {len(fields)} fields, {off} tokens", flush=True)
    return np.stack(cols, 1), off


class FM(torch.nn.Module):
    def __init__(self, n_tok: int, k: int) -> None:
        super().__init__()
        self.w = torch.nn.Embedding(n_tok, 1)
        self.v = torch.nn.Embedding(n_tok, k)
        torch.nn.init.zeros_(self.w.weight)
        torch.nn.init.normal_(self.v.weight, std=0.01)
        self.b = torch.nn.Parameter(torch.tensor([-4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v(x)
        inter = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1)
        return self.b + self.w(x).sum((1, 2)) + inter


def fit_predict(xtr, ytr, xev: list, n_tok: int, a: argparse.Namespace, seed: int) -> list:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = FM(n_tok, a.k).cuda()
    opt = torch.optim.AdamW([{"params": m.w.parameters(), "weight_decay": a.wd},
                            {"params": m.v.parameters(), "weight_decay": a.wd},
                            {"params": [m.b], "weight_decay": 0}], lr=a.lr)
    xt = torch.as_tensor(xtr).cuda()
    yt = torch.as_tensor(ytr, dtype=torch.float32).cuda()
    n = len(ytr)
    steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    step = 0
    for _ in range(a.epochs):
        perm = torch.as_tensor(rng.permutation(n)).cuda()
        for b in range(0, n - a.bs + 1, a.bs):
            idx = perm[b : b + a.bs]
            xb = xt[idx]
            if a.fdrop:  # field dropout: replace with a per-batch null token (index 0 of w/v? use mask)
                keep = (torch.rand(xb.shape, device=xb.device) > a.fdrop).float()
                v = m.v(xb) * keep[..., None]
                inter = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1)
                out = (m.b + (m.w(xb)[..., 0] * keep).sum(1) + inter) / 1.0
            else:
                out = m(xb)
            # L2 on the embeddings actually used in the batch (standard FM regularisation)
            reg = a.l2v * m.v(xb).pow(2).sum() / a.bs + a.l2w * m.w(xb).pow(2).sum() / a.bs
            loss = bce(out, yt[idx]) + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
    m.eval()
    outs = []
    with torch.no_grad():
        for xe in xev:
            xe_t = torch.as_tensor(xe).cuda()
            outs.append(torch.cat([m(xe_t[i : i + 65536]) for i in range(0, len(xe_t), 65536)])
                        .cpu().numpy())
            del xe_t
    del m, opt, xt, yt
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l2v", type=float, default=1e-4)
    ap.add_argument("--l2w", type=float, default=1e-5)
    ap.add_argument("--min_cnt", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop", nargs="*", default=[], help="token families: full ipd1 last2")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW decay on all tokens every step")
    ap.add_argument("--fdrop", type=float, default=0.0, help="field dropout during training")
    a = ap.parse_args()
    name = f"fm_k{a.k}{a.tag}"
    out = RESULTS_DIR / f"_{name}"
    out.mkdir(exist_ok=True)
    tr = pd.read_csv(DATA_DIR / "train.csv", dtype=str)
    te = pd.read_csv(DATA_DIR / "test.csv", dtype=str)
    tr["id"] = tr["id"].astype(np.int64)
    te["id"] = te["id"].astype(np.int64)
    tr, te = tr.sort_values("id").reset_index(drop=True), te.sort_values("id").reset_index(drop=True)
    y = tr[TARGET].astype(float).astype(int).to_numpy()
    folds = fold_assignment(y)
    X, n_tok = tokens(pd.concat([tr[RAW_COLS], te[RAW_COLS]], ignore_index=True), a.min_cnt, tuple(a.drop))
    Xtr, Xte = X[: len(tr)], X[len(tr):]
    assert NUM_COLS
    for k in a.folds:
        t0 = time.time()
        trn, va = folds != k, folds == k
        pv, pt = np.zeros(va.sum()), np.zeros(len(Xte))
        for s in range(a.seeds):
            ev = [Xtr[va]] if a.no_test else [Xtr[va], Xte]
            res = fit_predict(Xtr[trn], y[trn], ev, n_tok, a, 100 * k + s)
            pv += res[0] / a.seeds
            if not a.no_test:
                pt += res[1] / a.seeds
            print(f"  fold {k} seed {s}: auc {roc_auc_score(y[va], res[0]):.5f}", flush=True)
        print(f"  fold {k}: auc {roc_auc_score(y[va], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out / f"f{k}_val.npy", pv)
        if not a.no_test:
            np.save(out / f"f{k}_test.npy", pt)


if __name__ == "__main__":
    main()
