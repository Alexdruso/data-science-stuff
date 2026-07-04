"""Strong single LGBM: GENUINE low-lr / high-rounds regime — PS S6E6.

v1 (lr=0.015 but num_leaves=173 carried from best_params) bottomed at best_iter
~1000 and 0.9652 — but that was a tree-capacity artifact, not saturation: fat
173-leaf trees overfit validation logloss in a few hundred rounds regardless of
lr. The genuine regime is low lr + SHALLOW trees (num_leaves ~63) so the ensemble
builds gradually over many thousands of rounds.

Two corrections vs v1:
  1. num_leaves=63 (shallow) + lr=0.02 + 20k rounds + patience 200. If best_iter
     is still ~1000, model-tuning really is exhausted; expect several thousand.
  2. Early-stop on BALANCED ACCURACY (custom eval), not multi_logloss — we train a
     class-weighted objective but the score is unweighted balanced acc; stopping on
     logloss is a mismatch on the exact axis we're judged on.

CV is a proven LB proxy here (adversarial AUC=0.5), so a real CV lift transfers.
Gate argmax vs 0.9654 incumbent.

Run:  python src/train_lgbm_strong.py
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

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5

LR = 0.02
NUM_LEAVES = 63       # shallow — the v1 fix: build gradually over many rounds
N_ESTIMATORS = 20000
PATIENCE = 200
RUN = "lgbm_strong_v2"


def balanced_acc_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    """LGBM custom eval: balanced accuracy (higher better). y_pred is
    (n_samples, n_classes) in the sklearn multiclass wrapper."""
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(len(y_true), -1)
    return "bal_acc", float(balanced_accuracy_score(y_true, y_pred.argmax(axis=1))), True


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    params: dict[str, object] = {
        "objective": "multiclass", "num_class": 3,
        "metric": "None",  # disable builtin so early stopping watches bal_acc only
        "class_weight": "balanced", "random_state": 42, "verbose": -1,
        "n_jobs": n_jobs, "device_type": dev_type,
    }
    bp_path = RESULTS_DIR / "best_params.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text())
        for k in ("learning_rate", "num_leaves"):
            bp.pop(k, None)  # overridden — v1's fat num_leaves was the bug
        params.update(bp)
        print(f"Loaded shape params (minus lr/num_leaves) from {bp_path}")
    params.update(learning_rate=LR, num_leaves=NUM_LEAVES, n_estimators=N_ESTIMATORS)

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
    print(f"X {X.shape} | lr={LR} num_leaves={NUM_LEAVES} n_est={N_ESTIMATORS} "
          f"patience={PATIENCE} early-stop=bal_acc device={dev_type}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        m = LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
              eval_metric=balanced_acc_eval,
              callbacks=[early_stopping(PATIENCE, first_metric_only=True, verbose=False),
                         log_evaluation(0)])
        oof[vai] = m.predict_proba(X.iloc[vai])
        test_proba += m.predict_proba(X_test) / N_FOLDS
        best_iters.append(m.best_iteration_ or N_ESTIMATORS)
        score = float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1)))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}  (best_iter {best_iters[-1]})")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [incumbent 0.9654]  "
          f"mean best_iter {int(np.mean(best_iters))}")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")

    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}  [incumbent 0.9657]")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels)
    print(f"Saved oof/test/submission → {RUN}")


if __name__ == "__main__":
    main()
