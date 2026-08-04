"""Surface `s6` — `s1` restricted to the columns that carry signal.

Seven of the twelve raw columns are noise on this target *and* unreconstructable
from their siblings:

    age 0.5032   sleep_hours 0.5289   notifications_per_day 0.4926
    app_opens_per_day 0.5413   gender / stress_level / academic_work_impact ~0.50

(single-feature AUC; OLS R^2 from siblings 0.000-0.005 for the four numerics).
All the signal sits in `daily`, `social_media`, `weekend`, `work_study`,
`gaming`.

With `colsample_bytree=0.8`, carrying 7/12 noise raw columns — plus the four v1
ratios built on top of them — dilutes every split's candidate set. This arm
tests whether removing them pays.

Honest prior: **small**. GBDTs are fairly robust to noise features, and a
single-feature AUC of 0.50 does not prove a column is useless in *interaction*
(the fitted logistic gave `sleep_hours` a small but non-zero +0.104 coefficient,
and `app_opens_per_day` reads 0.5413, marginally above chance). The bar is the
same pre-registered one as the rest of the sweep and does not move.
"""

import polars as pl

from features_v2 import build_features_v2

#: Raw columns with single-feature AUC < 0.55 and no reconstructive value.
NOISE_RAW: list[str] = [
    "age",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "gender",
    "stress_level",
    "academic_work_impact",
]

#: v1/v2 derived columns whose inputs are entirely inside `NOISE_RAW`, or which
#: mix a signal column with a noise one (so their variation is noise-dominated).
NOISE_DERIVED: list[str] = [
    "stress_ord",
    "notifications_per_open",
    "minutes_per_open",
    "screen_share_of_awake",
    "free_hours",
]


def build_features_signal(df: pl.DataFrame) -> pl.DataFrame:
    """`build_features_v2` minus the noise columns. Row order is unchanged."""
    out = build_features_v2(df)
    drop = [c for c in (*NOISE_RAW, *NOISE_DERIVED) if c in out.columns]
    return out.drop(drop)
