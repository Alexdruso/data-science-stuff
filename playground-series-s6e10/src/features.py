"""Feature engineering for PS S6E10 — Predicting Airline Satisfaction.

`build_features()` owns the row-order invariant: it sorts by SORT_KEY FIRST,
then derives columns. Every `results/oof_*.npy` / `results/test_*.npy` array
is stored in that order, so any script that loads `y` or `test_ids` must go
through this function.

The raw CSV headers carry spaces and slashes ("Departure/Arrival time
convenient"); `build_features` renames them to snake_case so every downstream
script, model and feature-importance dump sees clean names. The target is
coerced to 0/1 Int8 whatever form it ships in (bool in the original dataset;
the synthetic split may use strings).

v1 is deliberately lean: raw columns plus a handful of domain aggregates.
In the original dataset a 0 on the 0-5 service ratings means "not
applicable", not "terrible" — kept as-is (trees split on it) and counted in
`n_zero_ratings`. LightGBM handles the NaN arrival delays natively.
"""

import re

import polars as pl

TARGET = "satisfaction"

# The dataset's only key. Sorting by it is a no-op on the shipped CSVs, but
# the invariant machinery reads this line — do not remove it.
SORT_KEY: list[str] = ["id"]

# Columns that are never model inputs.
EXCLUDE_COLS: frozenset[str] = frozenset({"id", TARGET})

RAW_CATEGORICAL_COLS: list[str] = [
    "gender",
    "customer_type",
    "type_of_travel",
    "class",
]

# 0-5 survey ratings (0 = not applicable in the original dataset).
RATING_COLS: list[str] = [
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "cleanliness",
]

RAW_NUMERIC_COLS: list[str] = [
    "age",
    "flight_distance",
    "departure_delay_in_minutes",
    "arrival_delay_in_minutes",
    *RATING_COLS,
]

# Accepted spellings of the positive class, lower-cased.
_POSITIVE_LABELS: frozenset[str] = frozenset({"true", "1", "satisfied"})


def snake_case(name: str) -> str:
    """'Departure/Arrival time convenient' -> 'departure_arrival_time_convenient'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def encode_target(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce TARGET to Int8 0/1; no-op if the frame has no target (test)."""
    if TARGET not in df.columns:
        return df
    col = pl.col(TARGET).cast(pl.String).str.to_lowercase()
    return df.with_columns(
        col.is_in(list(_POSITIVE_LABELS)).cast(pl.Int8).alias(TARGET)
    )


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by SORT_KEY, then add derived columns. Returns a Polars frame."""
    df = df.rename({c: snake_case(c) for c in df.columns})
    return derive_v1(encode_target(df.sort(SORT_KEY)))


def derive_v1(df: pl.DataFrame) -> pl.DataFrame:
    """The v1 derived columns, WITHOUT sorting.

    Kept separate from `build_features` so fold-level code can recompute the
    derived columns on a row subset whose order must not change.
    `build_features` remains the only entry point that sorts.
    """
    ratings = [pl.col(c) for c in RATING_COLS]
    dep = pl.col("departure_delay_in_minutes")
    arr = pl.col("arrival_delay_in_minutes")
    return df.with_columns(
        [
            # --- survey aggregates ---
            pl.mean_horizontal(ratings).alias("rating_mean"),
            pl.min_horizontal(ratings).alias("rating_min"),
            pl.max_horizontal(ratings).alias("rating_max"),
            pl.sum_horizontal([(r == 0).cast(pl.Int8) for r in ratings]).alias(
                "n_zero_ratings"
            ),
            pl.sum_horizontal([(r >= 4).cast(pl.Int8) for r in ratings]).alias(
                "n_high_ratings"
            ),
            # --- delays ---
            (arr - dep).alias("delay_recovered"),
            pl.max_horizontal([dep, arr]).alias("delay_max"),
            (dep / (pl.col("flight_distance") + 1.0)).alias("delay_per_mile"),
        ]
    )
