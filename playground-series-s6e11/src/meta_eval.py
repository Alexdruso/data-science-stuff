"""Compare meta-learners over the same leg logits (outer 5-fold, seed 7, same as stack.py).

    python src/meta_eval.py LEG1 LEG2 ... [--methods lr lgbm lgbm_raw] [--threads 4]
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import CAT_COLS, RAW_COLS, TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["lr", "lgbm", "lgbm_raw"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    X = np.column_stack([logit(np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel())
                         for n in a.names])
    raw = tr.select(RAW_COLS).with_columns(
        [pl.col(c).cast(pl.Categorical).to_physical() for c in CAT_COLS]).to_numpy().astype(float)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
    for meth in a.methods:
        oof = np.zeros(len(y))
        for trn, val in folds:
            if meth == "lr":
                m = LogisticRegression(max_iter=3000).fit(X[trn], y[trn])
                oof[val] = m.decision_function(X[val])
                continue
            Z = X if meth == "lgbm" else np.column_stack([X, raw])
            p = dict(objective="binary", learning_rate=0.02, num_leaves=4, max_depth=2,
                     min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=a.threads)
            # inner split of the outer-train part for early stopping
            rng = np.random.default_rng(0)
            es = rng.random(len(trn)) < 0.15
            dtr = lgb.Dataset(Z[trn[~es]], y[trn[~es]])
            dva = lgb.Dataset(Z[trn[es]], y[trn[es]])
            b = lgb.train(p, dtr, 4000, valid_sets=[dva], callbacks=[
                lgb.early_stopping(200, verbose=False)], feval=None)
            oof[val] = b.predict(Z[val], raw_score=True)
            print(f"    {meth} best_iter {b.best_iteration}", flush=True)
        print(f"{meth:10s} OOF AUC {roc_auc_score(y, oof):.5f}", flush=True)


if __name__ == "__main__":
    main()
