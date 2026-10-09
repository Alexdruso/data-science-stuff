"""Residual LGBM on top of the LR-on-logits pool stack, fully nested (10-09).

    python src/stack_resid.py [--ours LEG ...] [--C 0.1]

Outer fold k (base folds): LR stack fit on folds != k gives the base logit for fold k.
Training rows of the residual LGBM need base logits that never saw their own labels, so
they come from an inner 4-fold LR inside folds != k. The LGBM starts from that logit
(init_score) and may only use raw context features + the top pool logits; it early-stops
on a 10% slice of the training rows. Reports LR-only vs LR+residual OOF AUC per fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def lr_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xs: list[np.ndarray], C: float) -> list[np.ndarray]:
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return [lr.decision_function(sc.transform(X)) for X in Xs]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--ntop", type=int, default=20)
    a = p.parse_args()

    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = tr[TARGET].to_numpy()
    ctx_cols = [c for c in tr.columns if c not in (TARGET, "id")]
    ctx = tr.select(ctx_cols).to_pandas()
    for c in ctx.columns:
        if ctx[c].dtype == object or str(ctx[c].dtype).startswith("str"):
            ctx[c] = ctx[c].astype("category")
    names, O, _ = load_pairs(a.ours, [str(DATA_DIR / "public_oof" / "_pool")])
    print(f"{len(names)} members, {len(ctx_cols)} context cols", flush=True)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(O, y))
    base, resid = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr_i, va_i) in enumerate(folds):
        (base[va_i],) = lr_fit_predict(O[tr_i], y[tr_i], [O[va_i]], a.C)
        inner = np.zeros(len(tr_i))
        for itr, iva in StratifiedKFold(4, shuffle=True, random_state=k).split(tr_i, y[tr_i]):
            (inner[iva],) = lr_fit_predict(O[tr_i][itr], y[tr_i][itr], [O[tr_i][iva]], a.C)
        # top members by |corr| with the inner residual direction: just use solo AUC rank
        solo = np.array([roc_auc_score(y[tr_i], O[tr_i, j]) for j in range(O.shape[1])])
        top = np.argsort(-solo)[: a.ntop]
        X = ctx.copy()
        for j in top:
            X[f"m{j}"] = O[:, j]
        X["base"] = 0.0
        X.loc[X.index[tr_i], "base"] = inner
        X.loc[X.index[va_i], "base"] = base[va_i]
        rng = np.random.default_rng(k)
        es = rng.random(len(tr_i)) < 0.1
        fit_i, es_i = tr_i[~es], tr_i[es]
        dtr = lgb.Dataset(X.iloc[fit_i], y[fit_i], init_score=X["base"].to_numpy()[fit_i])
        des = lgb.Dataset(X.iloc[es_i], y[es_i], init_score=X["base"].to_numpy()[es_i], reference=dtr)
        params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
                      min_data_in_leaf=2000, feature_fraction=0.5, lambda_l2=10.0,
                      max_bin=63, verbose=-1, num_threads=10, seed=k)
        m = lgb.train(params, dtr, 3000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        resid[va_i] = base[va_i] + m.predict(X.iloc[va_i], raw_score=True)
        print(f"fold {k}: LR {roc_auc_score(y[va_i], base[va_i]):.6f}  "
              f"LR+resid {roc_auc_score(y[va_i], resid[va_i]):.6f}  iters {m.best_iteration}",
              flush=True)
    print(f"OOF LR {roc_auc_score(y, base):.6f}  LR+resid {roc_auc_score(y, resid):.6f}")
    for w in (0.25, 0.5, 0.75):
        print(f"  shrunk w={w}: {roc_auc_score(y, base + w * (resid - base)):.6f}")


if __name__ == "__main__":
    main()
