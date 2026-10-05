"""Baseline LightGBM model for PS S6E10 — Predicting Airline Satisfaction.

Binary classification, metric ROC AUC. OOF/test arrays store the
positive-class probability (1-D, n rows) in `build_features()` sort order.
AUC is rank-based, so there is no threshold/decision layer here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features
from lgbm_device import get_lgbm_device

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

N_FOLDS = 5
_DEVICE_TYPE, _N_JOBS = get_lgbm_device()
LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": _N_JOBS,
    "device_type": _DEVICE_TYPE,
}


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
    print(f"Train: {train_pl.shape}   Test: {test_pl.shape}")

    # ── feature matrix ─────────────────────────────────────────────────────
    cat_cols = [
        c
        for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()
    for col in cat_cols:
        # Share the category set so LightGBM sees identical codes in both frames.
        levels = pd.api.types.union_categoricals(
            [train_pd[col].astype("category"), test_pd[col].astype("category")]
        ).categories
        train_pd[col] = pd.Categorical(train_pd[col], categories=levels)
        test_pd[col] = pd.Categorical(test_pd[col], categories=levels)

    X = train_pd[feature_cols]
    X_test = test_pd[feature_cols]
    y = train_pd[TARGET].to_numpy()
    test_ids = test_pd["id"].to_numpy()
    print(f"Features: {len(feature_cols)}   Categoricals: {cat_cols}")
    print(f"Positive rate: {y.mean():.4f}")

    params = load_params(RESULTS_DIR, LGBM_PARAMS, "best_params.json")
    tuned = (RESULTS_DIR / "best_params.json").exists()

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = LGBMClassifier(**params)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof_proba, test_proba, fold_scores = run_cv(
        X,
        y,
        X_test,
        fit_fold,
        n_splits=N_FOLDS,
        n_outputs=1,
        score_fn=fold_score,
    )
    for fold, score in enumerate(fold_scores, 1):
        print(f"  Fold {fold} AUC: {score:.5f}")

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.5f}")

    run_name = "lgbm_v1_tuned" if tuned else "lgbm_v1"
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")

    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof_proba)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test_proba)
    # Always keep oof_lgbm.npy / test_lgbm.npy pointing at the latest LGBM run.
    np.save(RESULTS_DIR / "oof_lgbm.npy", oof_proba)
    np.save(RESULTS_DIR / "test_lgbm.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    out_path = write_submission(
        SUBMISSIONS_DIR, f"{run_name}.csv", test_ids, TARGET, test_proba
    )
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
