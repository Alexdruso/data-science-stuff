"""LGBM with autoregressive 'overdue' feature for PS S6E5 — F1 Pit Stop Prediction.

Two-pass approach:
  Pass 1 (already done): oof_lgbm.npy from baseline.py
  Pass 2 (this script): compute overdue from pass-1 OOF, retrain with it, sequential test inference.

overdue_i = cumulative sum of pass-1 predicted pit probabilities for all laps 1..i-1
            within the same (Driver, Race, Year, Stint), reset to 0 at each new stint.

Sequential test inference: since test overdue depends on previous predictions, we process
test laps in (Driver, Race, Year, LapNumber) order, updating overdue lap by lap.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import DRIVER_COLS, build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params.json"
    base: dict[str, object] = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
        "device": "gpu",
    }
    if params_path.exists():
        import json
        with params_path.open() as f:
            base.update(json.load(f))
        print(f"Loaded tuned params from {params_path}")
    return base


def to_pandas(df: pl.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in cat_cols:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("category")
    return pdf


def compute_overdue(train_pl: pl.DataFrame, oof_preds: np.ndarray) -> pl.DataFrame:
    """Add overdue column: cumulative pass-1 OOF within (Driver, Race, Year, Stint), lag 1."""
    return (
        train_pl
        .with_columns(pl.Series("_oof", oof_preds))
        .with_columns(
            pl.col("_oof")
            .cum_sum()
            .shift(1)
            .fill_null(0.0)
            .over(["Driver", "Race", "Year", "Stint"])
            .alias("overdue")
        )
        .drop("_oof")
    )


def main() -> None:
    train_pl_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_pl_raw = pl.read_csv(DATA_DIR / "test.csv")

    train_pl = build_features(train_pl_raw)
    test_pl = build_features(test_pl_raw)
    train_pl = compute_group_features(train_pl_raw, train_pl)
    test_pl = compute_group_features(train_pl_raw, test_pl)

    # Pass 1 OOF already exists — compute overdue from it
    oof_pass1 = np.load(RESULTS_DIR / "oof_lgbm.npy")
    train_pl = compute_overdue(train_pl, oof_pass1)

    # Sanity check: overdue should be 0 at start of each stint
    overdue_vals = train_pl["overdue"].to_numpy()
    print(f"overdue: min={overdue_vals.min():.4f} max={overdue_vals.max():.4f} "
          f"mean={overdue_vals.mean():.4f}  zeros={( overdue_vals == 0).sum()}", flush=True)

    # Test gets a zero overdue placeholder (updated during sequential inference)
    test_pl = test_pl.with_columns(pl.lit(0.0).alias("overdue"))

    _exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in _exclude
    ]
    feature_cols = [c for c in train_pl.columns if c not in _exclude]
    print(f"Features: {len(feature_cols)}  (includes overdue)", flush=True)

    train = to_pandas(train_pl, cat_cols)
    test = to_pandas(test_pl, cat_cols)

    X = train[feature_cols]
    y = train[TARGET].to_numpy()
    X_test = test[feature_cols].copy()
    test_ids = test["id"].to_numpy()

    params = load_params()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X))
    fold_aucs: list[float] = []

    RESULTS_DIR.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}", flush=True)

        # Save fold model for sequential test inference
        model.booster_.save_model(str(RESULTS_DIR / f"lgbm_ar_fold{fold}.txt"))

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}", flush=True)

    # --- Sequential test inference ---
    print("Running sequential test inference ...", flush=True)
    fold_models = [
        lgb.Booster(model_file=str(RESULTS_DIR / f"lgbm_ar_fold{f}.txt"))
        for f in range(1, N_FOLDS + 1)
    ]

    test_proba = np.zeros(len(X_test))

    # test is sorted by (Driver, Race, Year, LapNumber) via build_features;
    # groupby with sort=True processes groups in alphabetical order, and within
    # each group rows remain in their original (LapNumber) order.
    for _, group in test.groupby(["Driver", "Race", "Year", "Stint"], sort=True):
        group_idx = group.index.to_numpy()
        overdue = 0.0
        for row_idx in group_idx:
            X_test.loc[row_idx, "overdue"] = overdue
            row = X_test.loc[[row_idx]]
            pred = float(np.mean([m.predict(row)[0] for m in fold_models]))
            test_proba[row_idx] = pred
            overdue += pred

    print("Sequential inference done.", flush=True)

    save_cv_result(RESULTS_DIR, "lgbm_ar_v1", fold_aucs, oof_auc)
    np.save(RESULTS_DIR / "oof_lgbm_ar.npy", oof_proba)
    np.save(RESULTS_DIR / "test_lgbm_ar.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}", flush=True)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "lgbm_ar_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
