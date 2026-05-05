"""Feature engineering for PS S6E5 — F1 Pit Stop Prediction."""

import polars as pl

TARGET = "PitNextLap"
_SMOOTH_ALPHA = 20.0

# Domain-knowledge ordinal: SOFT degrades fastest, WET is off-scale
COMPOUND_ORDINAL: dict[str, int] = {
    "SOFT": 3,
    "MEDIUM": 2,
    "HARD": 1,
    "INTERMEDIATE": 1,
    "WET": 0,
}


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    # Sort required for the rolling LapTime_Delta feature
    df = df.sort(["Driver", "Race", "Year", "LapNumber"])

    return df.with_columns(
        [
            # --- v1 features (kept) ---
            (pl.col("Year") == 2023).cast(pl.Int8).alias("is_2023"),
            (pl.col("TyreLife") ** 2).alias("TyreLife_sq"),
            # --- v2 features ---
            (pl.col("TyreLife") + 1).log(base=2.718281828).alias("TyreLife_log"),
            (
                pl.col("Compound")
                .replace(COMPOUND_ORDINAL, default=1)
                .cast(pl.Int8)
                .alias("compound_ord")
            ),
            (pl.col("RaceProgress") * pl.col("Stint")).alias("race_progress_x_stint"),
            (pl.col("Cumulative_Degradation") / (pl.col("TyreLife") + 1)).alias(
                "degradation_rate"
            ),
        ]
    ).with_columns(
        [
            (pl.col("TyreLife") * pl.col("compound_ord")).alias("tyre_life_x_compound"),
            (
                pl.col("LapTime_Delta")
                .rolling_mean(window_size=3, min_periods=1)
                .over(["Driver", "Race", "Year"])
                .alias("lap_time_delta_roll3")
            ),
            (
                pl.col("LapTime_Delta")
                .rolling_mean(window_size=7, min_periods=1)
                .over(["Driver", "Race", "Year"])
                .alias("lap_time_delta_roll7")
            ),
            (
                pl.col("LapTime (s)")
                .rolling_mean(window_size=5, min_periods=1)
                .over(["Driver", "Race", "Year"])
                .alias("lap_time_s_roll5")
            ),
            pl.col("LapTime_Delta").shift(1).over(["Driver", "Race", "Year"]).alias("lap_time_delta_lag1"),
            pl.col("LapTime_Delta").shift(2).over(["Driver", "Race", "Year"]).alias("lap_time_delta_lag2"),
            pl.col("LapTime_Delta").shift(3).over(["Driver", "Race", "Year"]).alias("lap_time_delta_lag3"),
        ]
    ).with_columns(
        [
            (pl.col("LapTime (s)") - pl.col("lap_time_s_roll5")).alias("lap_time_vs_roll5"),
            pl.col("lap_time_delta_lag1").fill_null(0),
            pl.col("lap_time_delta_lag2").fill_null(0),
            pl.col("lap_time_delta_lag3").fill_null(0),
        ]
    )


def _smoothed_rate(
    ref: pl.DataFrame,
    group_keys: list[str],
    global_rate: float,
    out_col: str,
) -> pl.DataFrame:
    """Laplace-smoothed mean(TARGET) per group, returned as a small lookup frame."""
    return (
        ref.group_by(group_keys)
        .agg(
            pl.col(TARGET).sum().alias("_sum"),
            pl.len().alias("_n"),
        )
        .with_columns(
            (
                (pl.col("_sum") + _SMOOTH_ALPHA * global_rate)
                / (pl.col("_n") + _SMOOTH_ALPHA)
            ).alias(out_col)
        )
        .select(group_keys + [out_col])
    )


def compute_group_features(
    train_df: pl.DataFrame,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Join pre-computed group statistics onto df.

    train_df is the raw training set (used as reference for all aggregates).
    df is the frame to enrich — may be train or test.
    """
    ref = train_df.filter(pl.col("Year") != 2023)

    global_rate = float(ref[TARGET].mean())
    global_tyre_median = float(ref.filter(pl.col(TARGET) == 1)["TyreLife"].median())

    # --- smoothed pit-rate aggregates ---
    driver_rate = _smoothed_rate(ref, ["Driver"], global_rate, "driver_pit_rate")
    driver_compound_rate = _smoothed_rate(
        ref, ["Driver", "Compound"], global_rate, "driver_compound_pit_rate"
    )
    race_compound_rate = _smoothed_rate(
        ref, ["Race", "Compound"], global_rate, "race_compound_pit_rate"
    )

    # --- median TyreLife at pit (plain median; fill nulls with global) ---
    pit_rows = ref.filter(pl.col(TARGET) == 1)

    driver_tyre = pit_rows.group_by("Driver").agg(
        pl.col("TyreLife").median().alias("driver_median_tyre_life_at_pit")
    )
    race_compound_tyre = pit_rows.group_by(["Race", "Compound"]).agg(
        pl.col("TyreLife").median().alias("race_compound_median_tyre_life_at_pit")
    )
    compound_tyre = pit_rows.group_by("Compound").agg(
        pl.col("TyreLife").median().alias("compound_typical_life")
    )

    return (
        df.join(driver_rate, on="Driver", how="left")
        .join(driver_compound_rate, on=["Driver", "Compound"], how="left")
        .join(race_compound_rate, on=["Race", "Compound"], how="left")
        .join(driver_tyre, on="Driver", how="left")
        .join(race_compound_tyre, on=["Race", "Compound"], how="left")
        .join(compound_tyre, on="Compound", how="left")
        .with_columns(
            [
                pl.col("driver_pit_rate").fill_null(global_rate),
                pl.col("driver_compound_pit_rate").fill_null(global_rate),
                pl.col("race_compound_pit_rate").fill_null(global_rate),
                pl.col("driver_median_tyre_life_at_pit").fill_null(global_tyre_median),
                pl.col("race_compound_median_tyre_life_at_pit").fill_null(
                    global_tyre_median
                ),
                pl.col("compound_typical_life").fill_null(global_tyre_median),
            ]
        )
        .with_columns(
            (pl.col("TyreLife") / pl.col("compound_typical_life")).alias("TyreLife_frac")
        )
        .drop("compound_typical_life")
    )
