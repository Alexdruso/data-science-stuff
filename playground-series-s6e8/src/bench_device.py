"""One-fold timing of LightGBM on CUDA vs CPU, to pick ONE device for the sweep.

LightGBM's CUDA and CPU histogram paths do not build identical trees, so a
paired arm-vs-arm delta measured across mixed devices carries a nuisance term
larger than the effect being measured. Pick one here and use it throughout.
"""

import sys
import time
from pathlib import Path

import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import LGBM_PARAMS, N_FOLDS
from features import EXCLUDE_COLS, TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, _, _ = competition_dirs(__file__)


def main() -> None:
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    train_pd = train.to_pandas()
    for col in feature_cols:
        if train_pd[col].dtype == object:
            train_pd[col] = train_pd[col].astype("category")
    x_all = train_pd[feature_cols]
    y = train_pd[TARGET].to_numpy()

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    tr_idx, va_idx = next(iter(folds.split(x_all, y)))
    x_tr, x_va = x_all.iloc[tr_idx], x_all.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    print(f"one fold: train {x_tr.shape}  val {x_va.shape}\n")

    for device, n_jobs in (("cuda", 1), ("cpu", 12)):
        params = dict(LGBM_PARAMS)
        params.update(device_type=device, n_jobs=n_jobs)
        try:
            start = time.perf_counter()
            model = LGBMClassifier(**params)
            model.fit(
                x_tr,
                y_tr,
                eval_set=[(x_va, y_va)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            pred = model.predict_proba(x_va)[:, 1]
            elapsed = time.perf_counter() - start
            print(
                f"{device:>5}  n_jobs={n_jobs:<3} fold time {elapsed:7.1f}s  "
                f"-> 5-fold ~{elapsed * N_FOLDS / 60:.1f} min   "
                f"AUC {roc_auc_score(y_va, pred):.5f}  "
                f"best_iter {model.best_iteration_}"
            )
        except Exception as exc:  # noqa: BLE001 - a device that fails is a result
            print(f"{device:>5}  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
