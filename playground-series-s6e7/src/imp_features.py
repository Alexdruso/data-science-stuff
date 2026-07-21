"""Driver-reconstruction feature block (Day-17 P1 escalation, knn_features pattern).

Fold-safe HGBR sleep_duration ~ other 12 raw cols, fit on fold-train rows with
observed sleep; block = [imp_sleep, sleep_completed] for train-full and test.
Cached per (tag, seed, fold) like the te blocks — `tag` must identify the
surface (imputer inputs depend on which rows are NaN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from train_common import Dataset
from zoo_common import CACHE_DIR, _kseg

DRIVER = "sleep_duration"


def imp_block_for_fold(
    ds: Dataset,
    tr_idx: NDArray[np.int64],
    seed: int,
    fold: int,
    tag: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"imp_{tag}{_kseg()}_s{seed}_f{fold}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["train"], z["test"], list(z["names"])

    other_cols = [c for c in ds.feature_cols if c != DRIVER]

    def prep(df: pd.DataFrame, cats: "dict[str, pd.Index] | None" = None):  # noqa: ANN202
        X = df[other_cols].copy()
        vocab = {}
        for col in ds.cat_cols:
            if col not in X.columns:
                continue
            X[col] = X[col].fillna("missing").astype(str)
            levels = cats[col] if cats else pd.Index(sorted(X[col].unique()))
            X[col] = pd.Categorical(X[col], categories=levels)
            vocab[col] = levels
        return X, vocab

    X_tr, vocab = prep(ds.train)
    X_te, _ = prep(ds.test, vocab)
    sleep_tr = ds.train[DRIVER].to_numpy(dtype=np.float64)
    sleep_te = ds.test[DRIVER].to_numpy(dtype=np.float64)
    obs = ~np.isnan(sleep_tr)
    fit_mask = np.zeros(len(sleep_tr), dtype=bool)
    fit_mask[tr_idx] = True
    fit_mask &= obs

    reg = HistGradientBoostingRegressor(
        max_iter=300,
        early_stopping=True,
        categorical_features="from_dtype",
        random_state=seed + fold * 100,
    )
    reg.fit(X_tr.loc[fit_mask], sleep_tr[fit_mask])
    imp_tr = reg.predict(X_tr).astype(np.float32)
    imp_te = reg.predict(X_te).astype(np.float32)
    train_out = np.stack(
        [imp_tr, np.where(np.isnan(sleep_tr), imp_tr, sleep_tr).astype(np.float32)],
        axis=1,
    )
    test_out = np.stack(
        [imp_te, np.where(np.isnan(sleep_te), imp_te, sleep_te).astype(np.float32)],
        axis=1,
    )
    names = ["imp_sleep", "sleep_completed"]
    np.savez_compressed(cache, train=train_out, test=test_out, names=np.array(names))
    return train_out, test_out, names
