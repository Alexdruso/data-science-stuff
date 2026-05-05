"""XGBoost 5-fold CV training for PS S6E5 — F1 Pit Stop Prediction."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5
FIXED_PARAMS: dict[str, object] = {
    "n_estimators": 1000,
    "device": "cuda",
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "verbosity": 0,
    "random_state": 42,
}
# Sensible defaults used when best_params_xgboost.json is absent
DEFAULT_PARAMS: dict[str, object] = {
    "max_depth": 9,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "gamma": 0.0,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params_xgboost.json"
    base: dict[str, object] = {**FIXED_PARAMS, **DEFAULT_PARAMS}
    if params_path.exists():
        with params_path.open() as f:
            base.update(json.load(f))
        print(f"Loaded tuned params from {params_path}")
    else:
        print("No tuned params found — using sensible defaults")
    return base


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_csv(DATA_DIR / "train.csv"), pl.read_csv(DATA_DIR / "test.csv")


def main() -> None:
    train_raw, test_pl = load_data()
    train_pl = build_features(train_raw)
    test_pl = build_features(test_pl)
    train_pl = compute_group_features(train_raw, train_pl)
    test_pl = compute_group_features(train_raw, test_pl)
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    cat_cols = [
        c
        for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train_pl.columns if c not in ("id", TARGET)]

    train = train_pl.to_pandas()
    test = test_pl.to_pandas()
    for col in cat_cols:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")

    X = train[feature_cols]
    y = train[TARGET].to_numpy()
    X_test = test[feature_cols]
    test_ids = test["id"].to_numpy()

    params = load_params()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}")

    save_cv_result(RESULTS_DIR, "xgboost_v1", fold_aucs, oof_auc)

    np.save(RESULTS_DIR / "oof_xgboost.npy", oof_proba)
    np.save(RESULTS_DIR / "test_xgboost.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "xgboost_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
