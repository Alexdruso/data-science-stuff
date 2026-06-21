"""A/B: does the low-z specialist column help the REGION-AWARE GBDT meta? (S6E6)

The specialist beats the full stack on low-z STAR/GAL (0.9374 vs 0.9337) but added NOTHING to
the LINEAR LR stack — because it's a REGIONAL expert (valid only z<0.25) and a global linear
weight can't gate its column by region. The GBDT meta has raw redshift and can route on it, so
it's the right consumer. Clean A/B: identical regularized LGBM meta, 3-seed honest 5-fold,
WITH vs WITHOUT the specialist column.

Run:  python src/test_spec_gbdt_meta.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte"]
SEEDS = [2024, 7, 13]
EPS = 1e-6


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS))


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())

    logit_cols = [f"{m}_{c}" for m in MODELS for c in "gqs"]
    Ztr = pd.DataFrame(np.hstack([logit(np.load(RESULTS_DIR / f"oof_{m}.npy")) for m in MODELS]),
                       columns=logit_cols)
    feat_cols = [c for c in tr.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in feat_cols if tr[c].dtype == pl.String]
    tr_feat = tr.select(feat_cols).to_pandas()
    for c in cat_cols:
        tr_feat[c] = tr_feat[c].astype("category")
    X_base = pd.concat([Ztr, tr_feat], axis=1)

    spec = np.load(RESULTS_DIR / "oof_lowz_spec.npy")
    spec_df = pd.DataFrame({"spec_star": logit(spec), "spec_gal": logit(1 - spec)})
    X_spec = pd.concat([X_base, spec_df], axis=1)

    params = dict(
        objective="multiclass", num_class=3, class_weight="balanced",
        n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        verbose=-1, n_jobs=n_jobs, device_type=dev_type,
    )

    def run(X):
        oof = np.zeros((len(y), 3))
        for seed in SEEDS:
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            for tri, vai in skf.split(X, y):
                m = LGBMClassifier(**{**params, "random_state": seed})
                m.fit(X.iloc[tri], y[tri], categorical_feature=cat_cols)
                oof[vai] += m.predict_proba(X.iloc[vai]) / len(SEEDS)
        pred = np.argmax(oof, axis=1)
        am = float(balanced_accuracy_score(y, pred))
        _, thr = optimize_thresholds(oof, y)
        rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
        return am, thr, rec

    base_am, base_thr, base_rec = run(X_base)
    print(f"GBDT meta WITHOUT spec: argmax {base_am:.5f}  thresh {base_thr:.5f}  "
          f"rec {dict(zip(le.classes_, base_rec.round(4)))}")
    spec_am, spec_thr, spec_rec = run(X_spec)
    print(f"GBDT meta WITH    spec: argmax {spec_am:.5f}  thresh {spec_thr:.5f}  "
          f"rec {dict(zip(le.classes_, spec_rec.round(4)))}")
    print(f"\ndelta argmax {spec_am - base_am:+.5f}  thresh {spec_thr - base_thr:+.5f}  "
          f"[incumbent GBDT-stack 0.96981]")


if __name__ == "__main__":
    main()
