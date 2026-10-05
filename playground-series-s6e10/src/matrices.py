"""Shared model matrices for PS S6E10 legs.

One place builds the pandas X / y / X_test every trainer fits on, so the legs
share identical feature columns and identical category codes. Rows come out in
`build_features()` order (the row-order invariant), and the categorical level
set is the union over every frame passed in — including the original dataset
when it is requested, so its rows encode with the same codes as the synthetic
ones.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, RAW_CATEGORICAL_COLS, TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, _, _ = competition_dirs(__file__)

ORIGINAL_FLAG = "is_original"


@dataclass
class Matrices:
    X: pd.DataFrame
    y: np.ndarray
    X_test: pd.DataFrame
    test_ids: np.ndarray
    feature_cols: list[str]
    cat_cols: list[str]
    # Original dataset, encoded identically; None unless requested.
    X_orig: pd.DataFrame | None = None
    y_orig: np.ndarray | None = None


def load_original() -> pl.DataFrame:
    """Original dataset through `build_features`, with negative ids.

    The original has no id column; negative ids keep them disjoint from the
    synthetic ids and give `build_features` a key to sort on.
    """
    raw = pl.read_csv(DATA_DIR / "original" / "data.csv")
    raw = raw.with_columns((-pl.int_range(1, raw.height + 1)).alias("id"))
    return build_features(raw)


def build_matrices(*, with_original: bool = False) -> Matrices:
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    frames = [train, test]
    if with_original:
        frames.append(load_original())

    # The flag is a column in every arm (constant 0 when no original rows are
    # used) so the augmented and control arms fit on the same feature set.
    frames = [
        f.with_columns(pl.lit(int(i == 2), dtype=pl.Int8).alias(ORIGINAL_FLAG))
        for i, f in enumerate(frames)
    ]
    feature_cols = [c for c in frames[0].columns if c not in EXCLUDE_COLS]
    pds = [f.to_pandas() for f in frames]
    for col in RAW_CATEGORICAL_COLS:
        levels = pd.api.types.union_categoricals(
            [p[col].astype("category") for p in pds]
        ).categories
        for p in pds:
            p[col] = pd.Categorical(p[col], categories=levels)

    m = Matrices(
        X=pds[0][feature_cols],
        y=pds[0][TARGET].to_numpy(),
        X_test=pds[1][feature_cols],
        test_ids=pds[1]["id"].to_numpy(),
        feature_cols=feature_cols,
        cat_cols=list(RAW_CATEGORICAL_COLS),
    )
    if with_original:
        m.X_orig = pds[2][feature_cols]
        m.y_orig = pds[2][TARGET].to_numpy()
    return m
