"""Fast sequential test inference for LGBM-AR — skips fold training.

Uses pre-saved lgbm_ar_fold{1..5}.txt models with num_threads=1 to avoid
the ~20ms/call thread-spawn overhead that makes single-row prediction slow.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import DRIVER_COLS, build_features, compute_group_features, compute_race_lap_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
TARGET = "PitNextLap"
N_FOLDS = 5


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")

    test_pl = build_features(test_raw)
    test_pl = compute_group_features(train_raw, test_pl)
    test_pl = compute_race_lap_features(test_pl)
    test_pl = test_pl.with_columns(pl.lit(0.0).alias("overdue"))

    _exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [c for c in test_pl.columns if test_pl[c].dtype == pl.String and c not in _exclude]
    feature_cols = [c for c in test_pl.columns if c not in _exclude]

    test = test_pl.to_pandas()
    for col in cat_cols:
        test[col] = test[col].astype("category")

    X_test = test[feature_cols].copy()
    test_ids = test["id"].to_numpy()

    fold_models = [
        lgb.Booster(model_file=str(RESULTS_DIR / f"lgbm_ar_fold{f}.txt"))
        for f in range(1, N_FOLDS + 1)
    ]
    print(f"Loaded {N_FOLDS} fold models. Features: {len(feature_cols)}")

    test_proba = np.zeros(len(X_test))
    overdue_col_pos = X_test.columns.get_loc("overdue")

    print("Running sequential test inference ...", flush=True)
    for _, group in test.groupby(["Driver", "Race", "Year", "Stint"], sort=True):
        group_idx = group.index.to_numpy()
        overdue = 0.0
        for row_idx in group_idx:
            X_test.iat[row_idx, overdue_col_pos] = overdue
            row = X_test.iloc[[row_idx]]
            pred = float(np.mean([m.predict(row, num_threads=1)[0] for m in fold_models]))
            test_proba[row_idx] = pred
            overdue += pred

    print("Sequential inference done.", flush=True)

    np.save(RESULTS_DIR / "test_lgbm_ar.npy", test_proba)
    print(f"test_lgbm_ar.npy saved → {RESULTS_DIR}")

    out_path = write_submission(SUBMISSIONS_DIR, "lgbm_ar_v1.csv", test_ids, TARGET, test_proba)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
