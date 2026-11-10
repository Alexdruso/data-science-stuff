"""Prototype memorisation probe: exact column matches between generated rows and original rows.

    python src/proto_match.py [--neg 339] -> results/proto_match.npz (train + test features)
"""

import argparse

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

COLS = None


def codes(frames: list[pl.DataFrame], cols: list[str]) -> list[np.ndarray]:
    out = [np.zeros((f.height, len(cols)), dtype=np.int32) for f in frames]
    for j, c in enumerate(cols):
        vals = []
        for f in frames:
            v = f[c].str.replace("True", "1").str.replace("False", "0")
            num = v.cast(pl.Float64, strict=False)
            vals.append(num.round(6).cast(pl.Utf8) if num.null_count() == 0 else v)
        uni = {v: i for i, v in enumerate(sorted(set().union(*[set(v.to_list()) for v in vals])))}
        for o, v in zip(out, vals):
            o[:, j] = np.array([uni[x] for x in v.to_list()], dtype=np.int32)
    return out


CAT = {"merchant_category", "card_type", "auth_method", "channel", "device_type"}


def maxmatch(X: np.ndarray, P: np.ndarray, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Per row: max #equal columns over prototypes, and #prototypes attaining >= max-1."""
    mx = np.zeros(len(X), dtype=np.int16)
    top = np.zeros(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        m = (X[i:i + chunk, None, :] == P[None, :, :]).sum(2)
        mx[i:i + chunk] = m.max(1)
        top[i:i + chunk] = np.sort(m, 1)[:, -5:].mean(1)
    return mx, top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=int, default=339)
    a = ap.parse_args()
    tr = pl.read_csv("data/train.csv", infer_schema=False).with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te = pl.read_csv("data/test.csv", infer_schema=False).with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    og = pl.read_csv("data/orig/credit_card_fraud_2026.csv", infer_schema=False)
    cols = [c for c in og.columns if c not in ("transaction_id", "is_fraud")]
    Xtr, Xte, Xog = codes([tr, te, og], cols)
    y = tr["is_fraud"].cast(pl.Float64).to_numpy().astype(int)
    yo = og["is_fraud"].cast(pl.Float64).to_numpy().astype(int)
    rng = np.random.default_rng(0)
    pos = Xog[yo == 1]
    neg = Xog[rng.choice(np.where(yo == 0)[0], a.neg, replace=False)]
    feats = {}
    for name, X in (("tr", Xtr), ("te", Xte)):
        mp, tp = maxmatch(X, pos)
        mn, tn = maxmatch(X, neg)
        feats[f"{name}_mp"], feats[f"{name}_tp"], feats[f"{name}_mn"], feats[f"{name}_tn"] = mp, tp, mn, tn
        if name == "tr":
            print("max match to fraud protos: AUC", round(roc_auc_score(y, mp), 4),
                  " top5 mean AUC", round(roc_auc_score(y, tp), 4))
            print("max match to nonfraud protos: AUC", round(roc_auc_score(y, mn), 4),
                  " top5 mean AUC", round(roc_auc_score(y, tn), 4))
            print("diff top5 AUC", round(roc_auc_score(y, tp - tn), 4))
            for v in range(int(mp.min()), int(mp.max()) + 1):
                s = mp == v
                if s.sum():
                    print(f"  maxmatch_fraud={v:2d}: n={s.sum():7d} fraud rate {y[s].mean():.4f}")
    np.savez("results/proto_match.npz", **feats)


if __name__ == "__main__":
    main()
