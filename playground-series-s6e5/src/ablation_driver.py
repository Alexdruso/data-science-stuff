"""Ablation: remove all Driver-identity features from LGBM v8.

Comparison baseline: baseline_lgbm_v8, OOF AUC = 0.9474 (Driver present).

Excluded features:
  - Driver              (raw categorical)
  - driver_pit_rate     (compute_group_features aggregate)
  - driver_compound_pit_rate
  - driver_median_tyre_life_at_pit
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_FOLDS = 5
FIXED_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
DRIVER_COLS: set[str] = {
    "Driver",
    "driver_pit_rate",
    "driver_compound_pit_rate",
    "driver_median_tyre_life_at_pit",
}


def load_params() -> dict[str, object]:
    params: dict[str, object] = dict(FIXED_PARAMS)
    path = RESULTS_DIR / "best_params.json"
    if path.exists():
        with path.open() as f:
            params.update(json.load(f))
        print(f"Loaded tuned params from {path}")
    return params


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train_pl = build_features(train_raw)
    train_pl = compute_group_features(train_raw, train_pl)

    exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in exclude
    ]
    feature_cols = [c for c in train_pl.columns if c not in exclude]

    print(f"Features used: {len(feature_cols)}  (dropped {len(DRIVER_COLS)} Driver cols)")
    print(f"Dropped: {sorted(DRIVER_COLS & set(train_pl.columns))}")

    train_pd = train_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")

    X = train_pd[feature_cols]
    y = train_pd[TARGET].to_numpy()

    params = load_params()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    fold_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_auc = float(roc_auc_score(y[va_idx], oof[va_idx]))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")

    oof_auc = float(roc_auc_score(y, oof))
    print(f"\nOOF AUC (no Driver): {oof_auc:.4f}")
    print(f"Baseline v8 (Driver): 0.9474")
    print(f"Delta:                {oof_auc - 0.9474:+.4f}")

    save_cv_result(RESULTS_DIR, "lgbm_no_driver", fold_aucs, oof_auc)


if __name__ == "__main__":
    main()
