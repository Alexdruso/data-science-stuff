"""Seed-bagged LGBM — the cheap, reliable variance-reduction bank (PS S6E6).

Fold scores swing 0.9644-0.9663 (~0.002 seed/fold variance); every model we've
built is single-seed. Averaging several seeds' probabilities is a CV-honest
+0.001-0.003 with no new ideas. Per fold we fit N_SEEDS LGBMs (tuned config,
different random_state) and average their val/test probabilities.

Run:  python src/train_lgbm_bag.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 5
SEEDS = [42, 43, 44, 45, 46]
RUN = "lgbm_bag"


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    params: dict[str, object] = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "n_estimators": 1000, "class_weight": "balanced", "verbose": -1,
        "n_jobs": n_jobs, "device_type": dev_type,
    }
    params.update(json.loads((RESULTS_DIR / "best_params.json").read_text()))

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")
        test_pd[col] = test_pd[col].astype("category")
    X, X_test = train_pd[feature_cols], test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"X {X.shape}  bagging {len(SEEDS)} seeds {SEEDS}  device {dev_type}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores: list[float] = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        v = np.zeros((len(vai), 3))
        for s in SEEDS:
            m = LGBMClassifier(**params, random_state=s)
            m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            v += m.predict_proba(X.iloc[vai]) / len(SEEDS)
            test_proba += m.predict_proba(X_test) / (len(SEEDS) * N_FOLDS)
        oof[vai] = v
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(v, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [single-seed lgbm 0.9654]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}  [incumbent 0.9657]")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
