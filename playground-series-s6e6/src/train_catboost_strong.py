"""CatBoost in the STRONG-BOOSTER regime — the untested base-strength lever (S6E6).

Our CatBoost is 0.9627 — anomalously BELOW our own LGBM (0.9654); cdeotte's CatBoost is
CV 0.9697. That 0.007 gap is a TUNING gap, not irreducibility. Two defects fixed here:

1. tune_catboost.py FIXED iterations=1000 and FLOORED lr≥0.02 → the strong regime (low lr,
   high rounds) was NEVER in the search space. We proved LGBM saturated via train_lgbm_strong
   (shallow + 20k rounds → best_iter still ~1000) but ran that test for LGBM ONLY. This is
   that test for CatBoost: lr=0.012, iterations=8000, early_stop=200.
2. eval_metric was plain "Accuracy" (UNWEIGHTED) → early-stopping peaked on the majority
   GALAXY class and undertrained the minority. With auto_class_weights="Balanced", weighted
   accuracy under inverse-freq weights == balanced accuracy exactly, so "Accuracy:use_weights=true"
   makes early-stopping target the real metric.

GATE: standalone argmax must clear ~0.9660 (beat our LGBM 0.9654 → confirms it WAS undertuned),
ideally ~0.966+; then re-stack must clear ~0.9670. Run name catboost_strong (does NOT clobber
catboost_v1 until it clears the gate). If it stays ~0.9627, the ceiling call is earned.

Run:  python src/train_catboost_strong.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5
RUN = "catboost_strong"

CB_PARAMS: dict[str, object] = {
    "iterations": 8000,
    "learning_rate": 0.012,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy:use_weights=true",  # == balanced acc under Balanced weights
    "auto_class_weights": "Balanced",
    "task_type": "GPU",
    "random_seed": 42,
    "verbose": 0,
}


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    cat_indices = [feature_cols.index(c) for c in cat_cols]
    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    X, X_test = train_pd[feature_cols], test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"X {X.shape}  classes {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        train_pool = Pool(X.iloc[tri], y[tri], cat_features=cat_indices)
        val_pool = Pool(X.iloc[vai], y[vai], cat_features=cat_indices)
        test_pool = Pool(X_test, cat_features=cat_indices)
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200)
        oof[vai] = model.predict_proba(val_pool)
        test_proba += model.predict_proba(test_pool) / N_FOLDS
        best_iters.append(int(model.get_best_iteration() or CB_PARAMS["iterations"]))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}  (best_iter {best_iters[-1]})")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [catboost_v1 0.9627; our LGBM 0.9654; deotte CB 0.9697]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
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
