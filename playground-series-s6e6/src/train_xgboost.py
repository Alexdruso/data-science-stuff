"""XGBoost training for PS S6E6 — Predicting Stellar Class.

5-fold stratified CV, GPU-accelerated. Saves OOF probabilities (n_train × 3)
and test probabilities (n_test × 3) for ensemble blending.
Handles class imbalance via inverse-frequency sample weights.
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
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("xgboost not installed — run: uv pip install xgboost") from e

from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

N_FOLDS = 5
XGB_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 50,  # XGBoost 2.0+: constructor param, not fit()
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

def make_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so minority classes get equal gradient mass."""
    counts = np.bincount(y).astype(float)
    w = 1.0 / counts[y]
    return (w / w.mean()).astype(np.float32)


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

    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()
    # XGBoost needs ordinal-encoded categoricals
    from sklearn.preprocessing import OrdinalEncoder
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_pd[cat_cols] = oe.fit_transform(train_pd[cat_cols])
        test_pd[cat_cols] = oe.transform(test_pd[cat_cols])

    X = train_pd[feature_cols]
    X_test = test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"Classes: {list(le.classes_)}")

    sample_weights = make_sample_weights(y)

    params = load_params(RESULTS_DIR, XGB_PARAMS, "best_params_xgboost.json")
    params["num_class"] = len(le.classes_)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_proba = model.predict_proba(X_val)
        oof_proba[val_idx] = val_proba
        test_proba += model.predict_proba(X_test) / N_FOLDS

        val_pred = np.argmax(val_proba, axis=1)
        score = float(balanced_accuracy_score(y_val, val_pred))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    oof_score = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))
    print(f"\nOOF balanced_acc (argmax): {oof_score:.4f}")

    threshold_weights, best_score = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    print(f"Threshold weights: {dict(zip(le.classes_, threshold_weights.round(4)))}")
    tw_path = RESULTS_DIR / "threshold_weights_xgboost.json"
    save_threshold_weights(threshold_weights, le.classes_.tolist(), tw_path)

    run_name = "xgb_v1"
    save_cv_result(RESULTS_DIR, run_name, fold_scores, best_score, metric_name="balanced_acc")

    np.save(RESULTS_DIR / "oof_xgboost.npy", oof_proba)
    np.save(RESULTS_DIR / "test_xgboost.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    test_pred_labels = le.inverse_transform(np.argmax(test_proba * threshold_weights, axis=1))
    out_path = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", test_ids, TARGET, test_pred_labels)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
