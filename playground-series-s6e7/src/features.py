"""Feature engineering for PS S6E7 — Predicting Student Health Risk.

Single source of truth for all feature transforms. Every training script and
notebook MUST import from here — never define transforms elsewhere.

Rules (see CLAUDE.md for full details):
- build_features() touches only raw input columns, never the target.
- All frames sorted by SORT_KEY so oof/test .npy arrays stay aligned.
- CLASSES fixes the label<->index mapping; oof/test .npy columns and the
  argmax->label decoding both key off this order. It equals sklearn/LGBM/XGB
  ``classes_`` (alphabetical) so integer 0/1/2 lines up; CatBoost's class order
  is verified explicitly at train time.
"""

import os

import polars as pl

TARGET = "health_condition"
SORT_KEY: list[str] = ["id"]

# Fixed class order (alphabetical == sklearn LabelEncoder order). OOF/test probability
# columns are stored in this order and argmax indexes decode through idx_to_label().
CLASSES: list[str] = ["at-risk", "fit", "unhealthy"]
label_to_idx: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
idx_to_label: dict[int, str] = dict(enumerate(CLASSES))

# Columns excluded from the feature matrix (id + target only).
EXCLUDE_COLS: frozenset[str] = frozenset({TARGET, "id"})

# Raw columns dropped from the model matrix — default: NONE (water_intake is KEPT).
# water_intake is pure target-noise, and dropping it looked like a free win — but the Day-2
# flip decomposition fingered the drop as the prime suspect for the v2 LB regression: in the
# near-zero-signal missing-driver region, removing even a noise column perturbs split
# tie-breaking and shifted the at-risk↔unhealthy boundary the champion had right (62% of the
# damaging flips, 88% missing-driver — see CLAUDE.md). S6E7_DROP_WATER=1 reproduces the
# water-dropped experimental runs; S6E7_KEEP_WATER stays honored for older commands.
DROP_COLS: frozenset[str] = (
    frozenset({"water_intake"})
    if os.environ.get("S6E7_DROP_WATER") == "1"
    and os.environ.get("S6E7_KEEP_WATER") != "1"
    else frozenset()
)

# Raw feature columns.
CAT_COLS: list[str] = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
NUM_COLS: list[str] = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by SORT_KEY and return. Safe to call on train or test.

    Baseline keeps engineering minimal (raw columns pass through) so the first
    models are a clean reference. Missing values are left as null — GBDTs handle
    NaN natively. Ordinal encoding / missingness indicators / target encoding are
    deferred (see CLAUDE.md "Next Steps"); adding them here later invalidates
    existing oof/test .npy and requires regenerating all models.

    Must never read or depend on TARGET — test has no labels.
    """
    return df.sort(SORT_KEY)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Model feature columns: everything except id, target, and DROP_COLS, in frame order."""
    return [c for c in df.columns if c not in (EXCLUDE_COLS | DROP_COLS)]
