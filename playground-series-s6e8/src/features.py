"""Feature engineering for PS S6E8 — Predicting Smartphone Addiction.

`build_features()` owns the row-order invariant: it sorts by SORT_KEY FIRST,
then derives columns. Every `results/oof_*.npy` / `results/test_*.npy` array
is stored in that order, so any script that loads `y` or `test_ids` must go
through this function.

v1 is deliberately lean: raw columns plus a handful of ratios that are
meaningful on this domain. LightGBM handles NaN natively, so no imputation
and no missingness indicators here — mask-derived features are a separate,
pre-registered experiment (see CLAUDE.md "Next steps"), and mixing them in
here would put every OOF array on a different surface.
"""

import polars as pl

TARGET = "addicted_label"

# The dataset's only key. Sorting by it is a no-op on the shipped CSVs, but
# the invariant machinery reads this line — do not remove it.
SORT_KEY: list[str] = ["id"]

# Columns that are never model inputs.
EXCLUDE_COLS: frozenset[str] = frozenset({"id", TARGET})

RAW_NUMERIC_COLS: list[str] = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

RAW_CATEGORICAL_COLS: list[str] = [
    "gender",
    "stress_level",
    "academic_work_impact",
]

# Domain ordinal — stress_level is the only ordered categorical.
STRESS_ORDINAL: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}

_EPS = 1e-6


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by SORT_KEY, then add derived columns. Returns a Polars frame."""
    return derive_v1(df.sort(SORT_KEY))


def derive_v1(df: pl.DataFrame) -> pl.DataFrame:
    """The v1 derived columns, WITHOUT sorting.

    Split out of `build_features` so fold-level code can recompute the derived
    columns after imputing the raw ones, on a row subset whose order must not
    change. `build_features` remains the only entry point that sorts, so the
    row-order invariant still has exactly one owner.
    """
    daily = pl.col("daily_screen_time_hours")
    return df.with_columns(
        [
            # --- composition of screen time ---
            (pl.col("social_media_hours") / (daily + _EPS)).alias("social_share"),
            (pl.col("gaming_hours") / (daily + _EPS)).alias("gaming_share"),
            (daily - pl.col("social_media_hours") - pl.col("gaming_hours")).alias(
                "other_screen_hours"
            ),
            (pl.col("weekend_screen_time") / (daily + _EPS)).alias("weekend_ratio"),
            # --- engagement intensity ---
            (pl.col("notifications_per_day") / (pl.col("app_opens_per_day") + _EPS)).alias(
                "notifications_per_open"
            ),
            ((daily * 60.0) / (pl.col("app_opens_per_day") + _EPS)).alias(
                "minutes_per_open"
            ),
            # --- time budget ---
            (daily / (24.0 - pl.col("sleep_hours") + _EPS)).alias("screen_share_of_awake"),
            (
                24.0
                - pl.col("sleep_hours")
                - pl.col("work_study_hours")
                - daily
            ).alias("free_hours"),
            # --- ordinal encoding (kept alongside the raw categorical) ---
            pl.col("stress_level")
            .replace_strict(STRESS_ORDINAL, default=None, return_dtype=pl.Int8)
            .alias("stress_ord"),
        ]
    )
