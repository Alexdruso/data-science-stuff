"""Baseline LightGBM model for PS S6E6 — Predicting Stellar Class."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
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
_DEVICE_TYPE, _N_JOBS = get_lgbm_device()
LGBM_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": _N_JOBS,
    "device_type": _DEVICE_TYPE,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params.json"
    base: dict[str, object] = dict(LGBM_PARAMS)
    if params_path.exists():
        with params_path.open() as f:
            tuned = json.load(f)
        base.update(tuned)
        print(f"Loaded tuned params from {params_path}")
    return base


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    return train, test


def main() -> None:
    train_pl_raw, test_pl_raw = load_data()

    # build_features() sorts both frames by SORT_KEY — all downstream .npy
    # arrays must be generated from frames produced by this call.
    train_pl = build_features(train_pl_raw)
    test_pl = build_features(test_pl_raw)

    train_pl = compute_group_features(train_pl_raw, train_pl)
    test_pl = compute_group_features(train_pl_raw, test_pl)

    print(f"Train: {train_pl.shape}   Test: {test_pl.shape}")

    # ── feature matrix ─────────────────────────────────────────────────────
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")
        test_pd[col] = test_pd[col].astype("category")

    X = train_pd[feature_cols]
    X_test = test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()

    # ── label encode target ────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"Classes: {list(le.classes_)}")  # e.g. ['GALAXY', 'QSO', 'STAR']

    params = load_params()
    params["num_class"] = len(le.classes_)
    tuned = (RESULTS_DIR / "best_params.json").exists()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
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

    run_name = "lgbm_v4_tuned" if tuned else "lgbm_v3_balanced"

    # ── threshold weight optimisation ──────────────────────────────────────
    threshold_weights, best_score = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    print(f"Threshold weights: {dict(zip(le.classes_, threshold_weights.round(4)))}")

    tw_path = RESULTS_DIR / f"threshold_weights_{run_name}.json"
    save_threshold_weights(threshold_weights, le.classes_.tolist(), tw_path)
    print(f"Threshold weights saved → {tw_path}")
    save_cv_result(RESULTS_DIR, run_name, fold_scores, best_score, metric_name="balanced_acc")

    # oof_lgbm.npy stores probabilities (n_train × n_classes) for ensemble blending
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof_proba)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test_proba)
    # Always keep oof_lgbm.npy / test_lgbm.npy pointing to the latest LGBM run for ensemble.py
    np.save(RESULTS_DIR / "oof_lgbm.npy", oof_proba)
    np.save(RESULTS_DIR / "test_lgbm.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    # ── submission ─────────────────────────────────────────────────────────
    test_pred = np.argmax(test_proba * threshold_weights, axis=1)
    test_pred_labels = le.inverse_transform(test_pred)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"{run_name}.csv"
    pd.DataFrame({"id": test_ids, TARGET: test_pred_labels}).to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
