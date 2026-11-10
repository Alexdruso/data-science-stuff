"""Residual boosting on top of a stack: LightGBM with init_score = stack OOF logit.

    python src/residual_boost.py pool_v1 [--feats raw digit] [--leaves 4] [--lr 0.02]

Outer StratifiedKFold(5, seed 11); inner early stopping on a 10% split of the outer-train rows.
Prints stack OOF AUC vs residual-boosted OOF AUC on identical rows.
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import BOOL_COLS, CAT_COLS, NUM_COLS, TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def digit_feats(df: pl.DataFrame) -> pl.DataFrame:
    """Digit keys of the float columns (first decimals, integer part mod 10)."""
    out = {}
    for c in NUM_COLS:
        v = df[c].cast(pl.Float64).to_numpy()
        if np.allclose(v, np.round(v)):
            continue
        out[f"{c}_d1"] = np.floor(np.abs(v) * 10 + 1e-9) % 10
        out[f"{c}_d2"] = np.floor(np.abs(v) * 100 + 1e-9) % 10
        out[f"{c}_i1"] = np.floor(np.abs(v) + 1e-9) % 10
    return pl.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack")
    ap.add_argument("--feats", nargs="*", default=["raw"])
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--min_leaf", type=int, default=500)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    p = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.stack, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = logit(p)
    parts = []
    if "raw" in a.feats:
        X = tr.select(NUM_COLS + CAT_COLS + BOOL_COLS).to_pandas()
        for c in CAT_COLS:
            X[c] = X[c].astype("category")
        for c in BOOL_COLS:
            X[c] = X[c].astype(float)
        parts.append(X)
    if "digit" in a.feats:
        parts.append(digit_feats(tr).to_pandas())
    import pandas as pd

    if "lm" in a.feats:
        lmf = {"lm_ep10": np.asarray(load_predictions(RESULTS_DIR, "lm_distilgpt2_ep10", ids)).ravel()}
        lmf["lm_ep10"] = np.log(lmf["lm_ep10"] / (1 - lmf["lm_ep10"]))
        lmf["lm_masa"] = np.load("ext/dl/s6e11-gpt2-target-first-oof/oof.npy")
        parts.append(pd.DataFrame(lmf))

    X = pd.concat(parts, axis=1)
    print(f"features {X.shape[1]}; stack OOF {roc_auc_score(y, base):.6f}")
    oof = np.zeros(len(y))
    params = {"objective": "binary", "learning_rate": a.lr, "num_leaves": a.leaves,
              "min_data_in_leaf": a.min_leaf, "lambda_l2": a.l2, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 3,
              "max_bin": 255}
    its = []
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=11).split(X, y):
        fit, es = train_test_split(trn, test_size=0.1, stratify=y[trn], random_state=0)
        dfit = lgb.Dataset(X.iloc[fit], y[fit], init_score=base[fit])
        des = lgb.Dataset(X.iloc[es], y[es], init_score=base[es], reference=dfit)
        m = lgb.train(params, dfit, 3000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        its.append(m.best_iteration)
        oof[va] = base[va] + m.predict(X.iloc[va], num_iteration=m.best_iteration, raw_score=True)
        print(f"  fold: iters {m.best_iteration}  base {roc_auc_score(y[va], base[va]):.6f} "
              f"-> {roc_auc_score(y[va], oof[va]):.6f}", flush=True)
    print(f"residual OOF {roc_auc_score(y, oof):.6f} (stack {roc_auc_score(y, base):.6f}) "
          f"iters {its}")


if __name__ == "__main__":
    main()
