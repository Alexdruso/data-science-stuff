"""Fold-safe naive-Bayes leg over exact written values (+ optional digit keys).

score = sum_c log P(v_c|1)/P(v_c|0), m-estimate smoothing toward the pooled value frequency.
    python src/nb_leg.py --keys raw|dig --m 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import NUM_COLS, RAW_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def keys(df: pl.DataFrame, mode: str) -> pl.DataFrame:
    out = [pl.col(c).alias(c) for c in RAW_COLS]
    if mode == "dig":
        for c in NUM_COLS:
            s = pl.col(c)
            if df[c].str.contains(r"\.").any():
                out += [s.str.split(".").list.get(0).alias(f"{c}_int"),
                        s.str.split(".").list.get(1, null_on_oob=True).fill_null("").alias(f"{c}_frac"),
                        (s.str.len_chars().cast(pl.Utf8) + "_" + s.str.slice(0, 2)).alias(f"{c}_l2")]
    return df.select(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="raw")
    ap.add_argument("--m", type=float, default=20.0)
    a = ap.parse_args()
    tr = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False).with_columns(
        pl.col("id").cast(pl.Int64)).sort("id")
    te = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False).with_columns(
        pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = ref[TARGET].to_numpy().astype(int)
    assert (ref["id"].to_numpy() == tr["id"].to_numpy()).all()
    folds = fold_assignment(y)
    Ktr, Kte = keys(tr, a.keys), keys(te, a.keys)
    oof, tst = np.zeros(len(y)), np.zeros(len(te))
    for k in range(1, 6):
        fit, va = folds != k, folds == k
        n1, n0 = y[fit].sum(), (1 - y[fit]).sum()
        for c in Ktr.columns:
            d = pl.DataFrame({"v": Ktr[c].to_numpy()[fit], "y": y[fit]}).group_by("v").agg(
                pl.col("y").sum().alias("c1"), pl.len().alias("n"))
            pi = n1 / (n1 + n0)
            r = (d["c1"].to_numpy() + a.m * pi) / (d["n"].to_numpy() + a.m)
            llr = np.log(r / (1 - r)) - np.log(pi / (1 - pi))
            lut = dict(zip(d["v"].to_list(), llr))
            oof[va] += np.array([lut.get(v, 0.0) for v in Ktr[c].to_numpy()[va]])
            tst += np.array([lut.get(v, 0.0) for v in Kte[c].to_numpy()]) / 5
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, oof)
    name = f"nb_{a.keys}_m{a.m:g}"
    print(name, auc, [round(roc_auc_score(y[folds == k], oof[folds == k]), 5) for k in range(1, 6)])
    save_predictions(RESULTS_DIR, name, oof=oof, test=tst, train_ids=tr["id"].to_numpy(),
                     test_ids=te["id"].to_numpy())
    save_cv_result(RESULTS_DIR, name, [], auc, metric_name="auc", params=vars(a))


if __name__ == "__main__":
    main()
