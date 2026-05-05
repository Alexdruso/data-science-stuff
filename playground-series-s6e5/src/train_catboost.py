"""CatBoost 5-fold CV training for PS S6E5 — F1 Pit Stop Prediction."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5
FIXED_PARAMS: dict[str, object] = {
    "iterations": 1000,
    "eval_metric": "AUC",
    "use_best_model": True,
    "early_stopping_rounds": 50,
    "task_type": "GPU",
    "verbose": 0,
    "random_seed": 42,
}
# Sensible defaults used when best_params_catboost.json is absent
DEFAULT_PARAMS: dict[str, object] = {
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "min_data_in_leaf": 50,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params_catboost.json"
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

    # CatBoost takes string columns as-is — no category dtype needed
    train = train_pl.to_pandas()
    test = test_pl.to_pandas()

    X = train[feature_cols]
    y = train[TARGET].to_numpy()
    X_test = test[feature_cols]
    test_ids = test["id"].to_numpy()

    params = load_params()
    params["cat_features"] = cat_cols

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}")

    save_cv_result(RESULTS_DIR, "catboost_v1", fold_aucs, oof_auc)

    np.save(RESULTS_DIR / "oof_catboost.npy", oof_proba)
    np.save(RESULTS_DIR / "test_catboost.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "catboost_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
