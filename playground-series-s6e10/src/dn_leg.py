"""PR-51: dependency-network generative classifier (own analogue of goodpjw2008's gen_v1).

    python src/dn_leg.py --folds 1          # fold-1 probe for screen_member.py
    python src/dn_leg.py --folds 1 2 3 4 5  # full member → results/oof_dn_leg.npy

For each discrete column c (13 ratings, 4 categoricals, age in 10 quantile bins) an XGBoost
predicts c from the other 20 raw columns AND the label y, fit on the fold's training rows only.
A validation/test row is scored at y=1 and at y=0; the member is the Bayes-style log-ratio
    s(x) = sum_c [log P(x_c | x_-c, y=1) - log P(x_c | x_-c, y=0)]
(pseudo-likelihood of a dependency network). Errors come from a generative route, unlike every
discriminative member in the pool. Folds = the exported tfm_fold{k}.npz splits (base folds).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RESULTS = Path(__file__).parent.parent / "results"
N_RAW = 21  # first 21 exported columns are the raw features (cats as codes)
SKIP = {"flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes"}


def discretise(col: str, tr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    if col == "age":
        edges = np.unique(np.quantile(tr, np.linspace(0.1, 0.9, 9)))
        return [np.searchsorted(edges, v, side="right") for v in (tr, *others)]
    vals = np.unique(tr)
    return [np.clip(np.searchsorted(vals, v), 0, len(vals) - 1) for v in (tr, *others)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--mode", choices=["dn", "chain"], default="dn")
    ap.add_argument("--route", action="store_true", help="chain: condition on fold-safe route TE too")
    ap.add_argument("--reverse", action="store_true", help="chain: reversed column order")
    ap.add_argument("--tag", default="chain_leg")
    a = ap.parse_args()
    if a.mode == "chain":
        chain(a)
        return
    out = RESULTS / "dn_leg"
    out.mkdir(exist_ok=True)
    for k in a.folds:
        z = np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)
        cols = list(z["cols"][:N_RAW])
        xtr, xva, xte = (z[n][:, :N_RAW].astype(np.float32) for n in ("x_tr", "x_va", "x_te"))
        ytr = z["y_tr"].astype(np.float32)
        sva, ste = np.zeros(len(xva)), np.zeros(len(xte))
        t0 = time.time()
        for j, c in enumerate(cols):
            if c in SKIP:
                continue
            ctr, cva, cte = discretise(c, xtr[:, j], xva[:, j], xte[:, j])
            nk = int(max(ctr.max(), cva.max(), cte.max())) + 1
            feat = [i for i in range(N_RAW) if i != j]
            Xtr = np.column_stack([xtr[:, feat], ytr])
            m = XGBClassifier(
                n_estimators=a.trees, learning_rate=0.1, max_depth=a.depth, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=k,
                objective="multi:softprob" if nk > 2 else "binary:logistic", verbosity=0,
            )
            m.fit(Xtr, ctr)
            for x, cc, s in ((xva, cva, sva), (xte, cte, ste)):
                lp = []
                for yv in (1.0, 0.0):
                    P = m.predict_proba(np.column_stack([x[:, feat], np.full(len(x), yv, np.float32)]))
                    p = P[np.arange(len(x)), np.clip(cc, 0, P.shape[1] - 1)]
                    lp.append(np.log(np.clip(p, 1e-6, 1)))
                s += lp[0] - lp[1]
            del m
            print(f"fold {k} {c:36s} nk={nk} running AUC {roc_auc_score(z['y_va'], sva):.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        print(f"dn fold {k}: AUC {roc_auc_score(z['y_va'], sva):.5f}", flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), sva]))
        np.save(out / f"f{k}_test.npy", ste)


def _xgb(nk: int, a: argparse.Namespace, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=a.trees, learning_rate=0.1, max_depth=a.depth, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=seed,
        objective="multi:softprob" if nk > 2 else "binary:logistic", verbosity=0,
    )


def _logp(m: XGBClassifier, cls: np.ndarray, x: np.ndarray, cc: np.ndarray, nk: int) -> np.ndarray:
    """log P(observed value) with classes absent from this class's training rows at 1e-6."""
    P = m.predict_proba(x)
    full = np.full((len(x), nk), 1e-6)
    full[:, cls.astype(int)] = P.reshape(len(x), -1) if len(cls) > 2 else np.column_stack([1 - P[:, -1], P[:, -1]])
    return np.log(np.clip(full[np.arange(len(x)), cc], 1e-6, 1))


def chain(a: argparse.Namespace) -> None:
    """PR-51b: exact Bayes chain. log-odds = logit P(y | fd, delays)
    + sum_j log P(x_j | x_<j, fd, delays, y=1) / P(x_j | x_<j, fd, delays, y=0),
    each conditional fit on that class's fold-training rows only."""
    out = RESULTS / a.tag
    out.mkdir(exist_ok=True)
    for k in a.folds:
        z = np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)
        cols = list(z["cols"])
        xtr, xva, xte = (z[n].astype(np.float32) for n in ("x_tr", "x_va", "x_te"))
        ytr = z["y_tr"].astype(int)
        cond = [cols.index(c) for c in SKIP]
        if a.route:
            cond += [i for i, c in enumerate(cols)
                     if c.startswith("TE_") or c in ("orig_fd_rate", "orig_fd_count", "cnt_flight_distance")]
        order = list(range(N_RAW))[::-1] if a.reverse else list(range(N_RAW))
        m = _xgb(2, a, k).fit(xtr[:, cond], ytr)
        pva, pte = (np.clip(m.predict_proba(x[:, cond])[:, 1], 1e-6, 1 - 1e-6) for x in (xva, xte))
        sva, ste = np.log(pva / (1 - pva)), np.log(pte / (1 - pte))
        print(f"fold {k} prior AUC {roc_auc_score(z['y_va'], sva):.5f}", flush=True)
        t0 = time.time()
        for j in order:
            c = cols[j]
            if c in SKIP:
                continue
            ctr, cva, cte = discretise(c, xtr[:, j], xva[:, j], xte[:, j])
            nk = int(max(ctr.max(), cva.max(), cte.max())) + 1
            for yv, sign in ((1, 1.0), (0, -1.0)):
                r = ytr == yv
                cls = np.unique(ctr[r])
                m = _xgb(len(cls), a, k).fit(xtr[r][:, cond], np.searchsorted(cls, ctr[r]))
                sva += sign * _logp(m, cls, xva[:, cond], cva, nk)
                ste += sign * _logp(m, cls, xte[:, cond], cte, nk)
                del m
            cond.append(j)
            print(f"fold {k} {c:36s} nk={nk} running AUC {roc_auc_score(z['y_va'], sva):.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        print(f"chain fold {k}: AUC {roc_auc_score(z['y_va'], sva):.5f}", flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), sva]))
        np.save(out / f"f{k}_test.npy", ste)


if __name__ == "__main__":
    main()
