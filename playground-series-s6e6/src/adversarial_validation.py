"""Adversarial validation: is playground TRAIN distributed like TEST? — PS S6E6.

We've optimized 5-fold CV for days but never checked whether CV is even a valid
proxy for the leaderboard. If train and test are drawn identically, a classifier
trying to tell them apart scores AUC ≈ 0.5 → CV is trustworthy, the LB gap is a
real modeling problem. If AUC ≫ 0.5, there's covariate shift → CV is optimistic,
and the lever is reweighting/selecting training rows toward the test distribution.

Self-contained diagnostic — no leaderboard feedback, no overfitting risk. Reports
the adversarial AUC + which features drive any shift, and (if shift exists) writes
per-train-row p(test) so a follow-up can importance-weight.

Run:  python src/adversarial_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, build_features, compute_group_features
from lgbm_device import get_lgbm_device

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    tr_pd, te_pd = train_pl.to_pandas(), test_pl.to_pandas()
    X = pd.concat([tr_pd[feature_cols], te_pd[feature_cols]], ignore_index=True)
    for col in cat_cols:
        X[col] = X[col].astype("category")
    y = np.concatenate([np.zeros(len(tr_pd)), np.ones(len(te_pd))]).astype(int)
    print(f"Adversarial set: {X.shape}  | train={len(tr_pd)} test={len(te_pd)}")

    dev_type, n_jobs = get_lgbm_device()
    params: dict[str, object] = {
        "objective": "binary", "metric": "auc", "n_estimators": 300,
        "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
        "colsample_bytree": 0.8, "class_weight": "balanced", "random_state": 42,
        "verbose": -1, "n_jobs": n_jobs, "device_type": dev_type,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    importances = np.zeros(len(feature_cols))
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        m = LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        importances += m.feature_importances_ / N_FOLDS
        print(f"  fold {fold} AUC: {roc_auc_score(y[vai], oof[vai]):.4f}")

    auc = roc_auc_score(y, oof)
    print(f"\n=== Adversarial AUC: {auc:.4f} ===")
    print("0.50 → train≈test (CV trustworthy); ≫0.50 → covariate shift (CV optimistic)\n")
    order = np.argsort(importances)[::-1]
    print("Top features distinguishing train vs test:")
    for i in order[:12]:
        print(f"  {feature_cols[i]:<18} {importances[i]:.0f}")

    # Per-train-row p(test): high = train row looks test-like (useful for weighting)
    p_test_train = oof[: len(tr_pd)]
    np.save(RESULTS_DIR / "adv_p_test_for_train.npy", p_test_train)
    print(f"\np(test) for train rows — mean {p_test_train.mean():.3f}, "
          f"[5,50,95]%: {np.round(np.percentile(p_test_train,[5,50,95]),3)}")
    print(f"Saved → {RESULTS_DIR / 'adv_p_test_for_train.npy'}")


if __name__ == "__main__":
    main()
