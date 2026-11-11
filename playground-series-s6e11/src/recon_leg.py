"""Label-free field reconstruction features + an XGBoost leg on them (PR-21).

For each numeric field, a LightGBM regressor predicts the field from the other 23 fields over
train + test rows (no label used), cross-fitted in 5 random folds over all rows. Features per field:
signed residual and |residual| / MAD-of-residuals. Then XGB d3 on raw (+ recon) for the chosen folds.

    python src/recon_leg.py --folds 1          # writes results/_recon/f1_val.npy (+ _raw control)
Recon features are cached in results/_recon/recon_feats.npy (train rows then test rows).
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
OUT = RESULTS_DIR / "_recon"
JOBS = 4


def matrix(df: pl.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c not in ("id", TARGET)]
    out = []
    for c in cols:
        s = df[c]
        if s.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
            s = s.cast(pl.Utf8).cast(pl.Categorical).to_physical()
        out.append(s.cast(pl.Float64).to_numpy())
    return np.column_stack(out), cols


def recon(X: np.ndarray, cols: list[str]) -> np.ndarray:
    feats = []
    for c in NUM_COLS:
        j = cols.index(c)
        others = np.delete(X, j, axis=1)
        pred = np.zeros(len(X))
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            m = lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.1, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, n_jobs=JOBS, verbose=-1)
            m.fit(others[tr], X[tr, j])
            pred[va] = m.predict(others[va])
        r = X[:, j] - pred
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        feats += [r, np.abs(r) / mad]
        print(f"  {c}: R2 {1 - r.var() / X[:, j].var():.3f}", flush=True)
    return np.column_stack(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    y = tr[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    Xa, cols = matrix(pl.concat([tr.drop(TARGET), te], how="vertical_relaxed"))
    cache = OUT / "recon_feats.npy"
    if cache.exists():
        R = np.load(cache)
    else:
        R = recon(Xa, cols)
        np.save(cache, R)
    n = len(tr)
    Xr = Xa[:n]
    Xf = np.hstack([Xr, R[:n]])
    for k in a.folds:
        trn, va = folds != k, folds == k
        for tag, X in (("raw", Xr), ("recon", Xf)):
            m = xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.05, max_bin=2048, subsample=0.8,
                                  colsample_bytree=0.5, n_jobs=JOBS, tree_method="hist", random_state=k)
            m.fit(X[trn], y[trn])
            p = m.predict_proba(X[va])[:, 1]
            d = OUT if tag == "recon" else RESULTS_DIR / "_recon_raw"
            d.mkdir(exist_ok=True)
            np.save(d / f"f{k}_val.npy", np.log(p / (1 - p)))
            print(f"fold {k} {tag}: auc {roc_auc_score(y[va], p):.5f}", flush=True)


if __name__ == "__main__":
    main()
