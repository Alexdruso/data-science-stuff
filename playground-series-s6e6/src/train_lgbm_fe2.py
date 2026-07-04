"""LGBM + cdeotte's redshift-interaction FE + SDSS17 class-prior features — PS S6E6.

Delta on top of lgbm_fe (add_deotte_features): adds add_sdss17_priors — per-quantile-
bin class RATES measured on the REAL SDSS17 data, smoothed toward the original global
prior (leak-safe external table). This is the last untried piece of cdeotte's XGB FE
(his XGB = CV 0.9669). Predict largely flat (the generator preserves P(class|features)).

GATE: argmax vs lgbm_fe 0.9649, AND low-z STAR recall (z in [0.01, 0.1] — the dominant
error cell) vs the lgbm_fe baseline. If it lands, re-stack; otherwise stop FE here and
move to the RealMLP + logit-adjust lever.

Run:  python src/train_lgbm_fe2.py
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
from features import (
    EXCLUDE_COLS,
    TARGET,
    add_deotte_features,
    add_sdss17_priors,
    build_features,
    compute_group_features,
)
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights
from train_lgbm_strong import balanced_acc_eval

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5
LR, NUM_LEAVES, N_ESTIMATORS, PATIENCE = 0.02, 63, 20000, 200
RUN = "lgbm_fe2"


def prep(raw: pl.DataFrame, train_raw: pl.DataFrame, orig: pl.DataFrame) -> pl.DataFrame:
    feat = add_deotte_features(compute_group_features(train_raw, build_features(raw)))
    return add_sdss17_priors(feat, orig)


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    params: dict[str, object] = {
        "objective": "multiclass", "num_class": 3, "metric": "None",
        "class_weight": "balanced", "random_state": 42, "verbose": -1,
        "n_jobs": n_jobs, "device_type": dev_type,
    }
    bp = json.loads((RESULTS_DIR / "best_params.json").read_text())
    for k in ("learning_rate", "num_leaves"):
        bp.pop(k, None)
    params.update(bp)
    params.update(learning_rate=LR, num_leaves=NUM_LEAVES, n_estimators=N_ESTIMATORS)

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    train_pl = prep(train_raw, train_raw, orig)
    test_pl = prep(test_raw, train_raw, orig)

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    print(f"feature count: {len(feature_cols)} (lgbm_fe was 38)")
    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")
        test_pd[col] = test_pd[col].astype("category")
    X, X_test = train_pd[feature_cols], test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()
    redshift = train_pd["redshift"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    star_idx = int(np.where(le.classes_ == "STAR")[0][0])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        m = LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
              eval_metric=balanced_acc_eval,
              callbacks=[early_stopping(PATIENCE, first_metric_only=True, verbose=False),
                         log_evaluation(0)])
        oof[vai] = m.predict_proba(X.iloc[vai])
        test_proba += m.predict_proba(X_test) / N_FOLDS
        best_iters.append(m.best_iteration_ or N_ESTIMATORS)
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}  (best_iter {best_iters[-1]})")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    # low-z STAR recall — the dominant error cell (advisor's targeted gate)
    lowz = (redshift >= 0.01) & (redshift <= 0.1)
    is_star = y == star_idx
    mask = lowz & is_star
    lowz_star_rec = float((pred[mask] == star_idx).mean()) if mask.any() else float("nan")
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [lgbm_fe 0.9649; deotte-xgb-FE 0.9669]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
    print(f"low-z STAR recall (z 0.01-0.1, n={int(mask.sum())}): {lowz_star_rec:.4f}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
