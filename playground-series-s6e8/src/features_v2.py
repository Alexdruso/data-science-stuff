"""Surface `s1` — constraint-derived features, no imputation.

This extends `features.build_features()` by CALLING it, so the `["id"]`
row-order invariant is inherited rather than re-implemented. `features.py`
stays frozen: `oof_lgbm_v1.npy` lives on that surface and surfaces do not mix.

The content here comes from one verified fact about the generating process:

    social_media_hours + gaming_hours + work_study_hours <= daily_screen_time_hours

holds in 100.0000% of complete train rows, with the residual attaining exactly
zero. So `daily = social + gaming + work + other`, `other >= 0`. Two things
follow that a GBDT cannot get on its own:

1. Because every term is non-negative, the *partial* sum of whichever of the
   three is observed is still a valid lower bound on `daily` — even on the
   13.9% of rows where `daily` itself is missing. `sgw_lb` is therefore an
   always-computable lower bound on the single best feature in the dataset.
2. The v1 features `other_screen_hours` and `free_hours` were mis-specified:
   `work_study_hours` is *inside* `daily`, not disjoint from it, so v1's
   `free_hours = 24 - sleep - work - daily` double-subtracts work. Corrected
   here. Expect ~nothing from the fix alone (a GBDT recovers linear
   combinations of present features cheaply) — it is corrected because it
   changes what the number means.
"""

import polars as pl

from features import (
    EXCLUDE_COLS,
    RAW_CATEGORICAL_COLS,
    RAW_NUMERIC_COLS,
    SORT_KEY,
    TARGET,
    derive_v1,
)

__all__ = [
    "BUDGET_PARTS",
    "EXCLUDE_COLS",
    "RAW_CATEGORICAL_COLS",
    "RAW_NUMERIC_COLS",
    "SCREEN_BLOCK",
    "SORT_KEY",
    "TARGET",
    "build_features_v2",
    "derive_v2",
]

#: The four columns whose missingness co-occurs (lift ~2.2) and whose values
#: are mutually predictive (daily R^2 0.81 from its siblings).
SCREEN_BLOCK: list[str] = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "weekend_screen_time",
]

#: The three non-negative components of `daily_screen_time_hours`.
BUDGET_PARTS: list[str] = [
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]

_EPS = 1e-6


def build_features_v2(df: pl.DataFrame) -> pl.DataFrame:
    """v1 features (incl. the SORT_KEY sort), plus constraint features."""
    return derive_v2(df.sort(SORT_KEY))


def derive_v2(df: pl.DataFrame) -> pl.DataFrame:
    """v1 + v2 derived columns, WITHOUT sorting — safe on a fold row subset."""
    out = derive_v1(df)

    daily = pl.col("daily_screen_time_hours")
    parts = [pl.col(c) for c in BUDGET_PARTS]
    # ignore_nulls: a partial sum of non-negative terms is still a lower bound.
    lower_bound = pl.sum_horizontal(*parts, ignore_nulls=True)
    n_obs = pl.sum_horizontal(
        *[pl.col(c).is_not_null().cast(pl.Int8) for c in BUDGET_PARTS]
    )

    return out.with_columns(
        [
            # --- the budget constraint ---
            lower_bound.alias("sgw_lb"),
            n_obs.alias("n_obs_sgw"),
            (daily - lower_bound).alias("daily_minus_lb"),
            (pl.col("social_media_hours") / (lower_bound + _EPS)).alias("social_of_lb"),
            # --- screen-block relations (weekend and daily are the same scale) ---
            (pl.col("weekend_screen_time") - daily).alias("weekend_minus_daily"),
            pl.sum_horizontal(
                *[pl.col(c).is_null().cast(pl.Int8) for c in SCREEN_BLOCK]
            ).alias("screen_block_nulls"),
            # --- corrected v1 features (work_study_hours is INSIDE daily) ---
            (daily - pl.sum_horizontal(*parts, ignore_nulls=False)).alias(
                "other_screen_hours"
            ),
            (24.0 - pl.col("sleep_hours") - daily).alias("free_hours"),
        ]
    )
