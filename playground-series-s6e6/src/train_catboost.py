"""CatBoost training for PS S6E6 — Predicting Stellar Class.

5-fold stratified CV, GPU-accelerated. Saves OOF probabilities (n_train × 3)
and test probabilities (n_test × 3) for ensemble blending.
auto_class_weights="Balanced" handles class imbalance natively.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as e:
    raise SystemExit("catboost not installed — run: uv pip install catboost") from e

from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

N_FOLDS = 5
CB_PARAMS: dict[str, object] = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "auto_class_weights": "Balanced",
    "task_type": "GPU",
    "random_seed": 42,
    "verbose": 0,
}

def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")

    train_pl = build_features(train_raw)
    test_pl = build_features(test_raw)
    train_pl = compute_group_features(train_raw, train_pl)
    test_pl = compute_group_features(train_raw, test_pl)

    print(f"Train: {train_pl.shape}   Test: {test_pl.shape}")

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    cat_indices = [feature_cols.index(c) for c in cat_cols]

    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    X = train_pd[feature_cols]
    X_test = test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"Classes: {list(le.classes_)}")

    params = load_params(RESULTS_DIR, CB_PARAMS, "best_params_catboost.json")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        test_pool = Pool(X_test, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
        )

        val_proba = model.predict_proba(val_pool)
        oof_proba[val_idx] = val_proba
        test_proba += model.predict_proba(test_pool) / N_FOLDS

        val_pred = np.argmax(val_proba, axis=1)
        score = float(balanced_accuracy_score(y_val, val_pred))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    oof_score = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))
    print(f"\nOOF balanced_acc (argmax): {oof_score:.4f}")

    threshold_weights, best_score = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    print(f"Threshold weights: {dict(zip(le.classes_, threshold_weights.round(4)))}")
    tw_path = RESULTS_DIR / "threshold_weights_catboost.json"
    save_threshold_weights(threshold_weights, le.classes_.tolist(), tw_path)

    run_name = "catboost_v1"
    save_cv_result(RESULTS_DIR, run_name, fold_scores, best_score, metric_name="balanced_acc")

    np.save(RESULTS_DIR / "oof_catboost.npy", oof_proba)
    np.save(RESULTS_DIR / "test_catboost.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    test_pred_labels = le.inverse_transform(np.argmax(test_proba * threshold_weights, axis=1))
    out_path = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", test_ids, TARGET, test_pred_labels)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
