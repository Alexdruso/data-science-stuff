"""Feature engineering for PS S6E11 — credit-card fraud (`is_fraud`, binary, AUC).

`build_features()` owns the row-order invariant: it sorts by SORT_KEY FIRST,
then derives columns. Every `results/oof_*.parquet` is keyed by `id`, but `y`
and ids must still come from this function.

The original dataset (uditjain13/credit-card-fraud-detection-2026, 20k rows)
lives in data/orig/ and is loaded by `load_original()` with negative ids so it
can be appended to train folds as extra rows (never scored).
"""

from pathlib import Path

import polars as pl

TARGET = "is_fraud"
SORT_KEY: list[str] = ["id"]
EXCLUDE_COLS: frozenset[str] = frozenset({"id", TARGET, "is_orig"})

CAT_COLS: list[str] = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]
BOOL_COLS: list[str] = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUM_COLS: list[str] = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
RAW_COLS: list[str] = NUM_COLS + CAT_COLS + BOOL_COLS


def load_original(data_dir: Path) -> pl.DataFrame:
    """Original rows with ids -1..-n, in the competition column layout."""
    orig = pl.read_csv(data_dir / "orig" / "credit_card_fraud_2026.csv")
    orig = orig.with_columns(
        (-pl.col("transaction_id")).cast(pl.Int64).alias("id"),
        pl.col(TARGET).cast(pl.Float64),
    ).drop("transaction_id")
    return orig.select(["id", *RAW_COLS, TARGET])


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by SORT_KEY, cast booleans to Int8 and the target to Int8."""
    df = df.sort(SORT_KEY)
    df = df.with_columns([pl.col(c).cast(pl.Int8) for c in BOOL_COLS])
    if TARGET in df.columns:
        df = df.with_columns(pl.col(TARGET).cast(pl.Int8))
    return df


FREQ_COLS: list[str] = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]


def add_extra(frames: list[pl.DataFrame], orig: pl.DataFrame) -> list[pl.DataFrame]:
    """Target-free value statistics: count of each value in train+test, and in orig.

    Uses only feature columns (no labels), so it is leak-free across folds.
    """
    full = pl.concat([f.select(FREQ_COLS) for f in frames])
    out = []
    for f in frames:
        for c in FREQ_COLS:
            cnt = full.group_by(c).agg(pl.len().cast(pl.Int32).alias(f"cnt_{c}"))
            ocnt = orig.group_by(c).agg(pl.len().cast(pl.Int32).alias(f"ocnt_{c}"))
            f = f.join(cnt, on=c, how="left").join(ocnt, on=c, how="left")
            f = f.with_columns(pl.col(f"ocnt_{c}").fill_null(0))
        f = f.with_columns(
            (pl.col("amount_usd") / (pl.col("account_balance_usd") + 1)).alias("amt_bal_ratio"),
        )
        out.append(f.sort(SORT_KEY))
    return out
