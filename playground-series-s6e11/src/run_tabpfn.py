"""TabPFN-3 leg (self-contained; run with /tmp/tabpfn_venv/bin/python).

    /tmp/tabpfn_venv/bin/python src/run_tabpfn.py --fold 1 --neg 90000          # fold pilot
    /tmp/tabpfn_venv/bin/python src/run_tabpfn.py --folds 1 2 3 4 5 --test       # full leg

Context per fold = ALL training-fold positives + a random `--neg` negatives (class prior shift
does not change the ranking). Features: raw columns (categoricals ordinal-coded) + digit keys of
the float columns. Saves results/_tabpfn{tag}/f{k}_val.npy (+ f{k}_test.npy); id-sorted order.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
V3_CKPT = Path.home() / ".cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt"
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]


def matrix(df: pl.DataFrame, cats: dict[str, list[str]]) -> np.ndarray:
    cols = []
    for c in df.columns:
        if c in ("id", "is_fraud"):
            continue
        if c in CAT:
            m = {v: i for i, v in enumerate(cats[c])}
            cols.append(np.array([m[v] for v in df[c].to_list()], dtype=np.float32))
        else:
            cols.append(df[c].cast(pl.Float64).to_numpy().astype(np.float32))
    for c in FLOATS:
        v = np.abs(df[c].cast(pl.Float64).to_numpy())
        cols += [np.floor(v * 10 + 1e-6) % 10, np.floor(v * 100 + 1e-6) % 10, np.floor(v + 1e-6) % 10]
    return np.column_stack(cols).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--neg", type=int, default=90000)
    ap.add_argument("--n_est", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tr = pl.read_csv(ROOT / "data/train.csv").sort("id")
    te = pl.read_csv(ROOT / "data/test.csv").sort("id")
    cats = {c: sorted(set(tr[c].to_list()) | set(te[c].to_list())) for c in CAT}
    X, Xt = matrix(tr, cats), matrix(te, cats)
    y = tr["is_fraud"].cast(pl.Float64).to_numpy().astype(int)
    cat_idx = [i for i, c in enumerate([c for c in tr.columns if c not in ("id", "is_fraud")])
               if c in CAT]
    folds = np.zeros(len(y), dtype=int)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        folds[va] = f
    out = ROOT / "results" / f"_tabpfn{a.tag}"
    out.mkdir(exist_ok=True)
    for k in a.folds:
        if (out / f"f{k}_val.npy").exists() and not a.n_val:
            continue
        rng = np.random.default_rng(100 + k)
        trn = np.where(folds != k)[0]
        pos, neg = trn[y[trn] == 1], trn[y[trn] == 0]
        ctx = np.concatenate([pos, rng.choice(neg, a.neg, replace=False)])
        va = np.where(folds == k)[0]
        if a.n_val:
            va = np.sort(rng.choice(va, a.n_val, replace=False))
        t0 = time.time()
        model = TabPFNClassifier(n_estimators=a.n_est, device="cuda", random_state=k,
                                 ignore_pretraining_limits=True, categorical_features_indices=cat_idx,
                                 memory_saving_mode=True, model_path=str(V3_CKPT))
        model.fit(X[ctx], y[ctx])
        pv = np.concatenate([model.predict_proba(X[va[i:i + a.chunk]])[:, 1]
                             for i in range(0, len(va), a.chunk)])
        print(f"fold {k}: ctx {len(ctx)} ({len(pos)} pos), val {len(va)} AUC "
              f"{roc_auc_score(y[va], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        if a.n_val:
            np.save(out / f"f{k}_valsub.npy", np.stack([va.astype(float), pv]))
        else:
            np.save(out / f"f{k}_val.npy", pv)
            if a.test:
                pt = np.concatenate([model.predict_proba(Xt[i:i + a.chunk])[:, 1]
                                     for i in range(0, len(Xt), a.chunk)])
                np.save(out / f"f{k}_test.npy", pt)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
