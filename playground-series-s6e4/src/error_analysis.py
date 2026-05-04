"""Analyse OOF errors from the formula baseline to guide feature engineering."""

from pathlib import Path

import numpy as np
import polars as pl

from formula import CLASSES, build_features, predict_logit

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

ALL_FEATURES = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Crop_Growth_Stage",
    "Mulching_Used",
    # unused by formula
    "Soil_Type",
    "Soil_pH",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Humidity",
    "Sunlight_Hours",
    "Crop_Type",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
    "Region",
]
NUMERIC_FEATURES = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Soil_pH",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Humidity",
    "Sunlight_Hours",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
CAT_FEATURES = [
    "Crop_Growth_Stage",
    "Mulching_Used",
    "Soil_Type",
    "Crop_Type",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Region",
]
THRESHOLDS = {
    "Soil_Moisture": 25.0,
    "Temperature_C": 30.0,
    "Rainfall_mm": 300.0,
    "Wind_Speed_kmh": 10.0,
}


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _value_counts(series: pl.Series, top_n: int = 10) -> None:
    vc = series.value_counts(sort=True).head(top_n)
    for row in vc.iter_rows():
        print(f"    {row[0]!s:<30} {row[1]}")


def main() -> None:
    train = pl.read_csv(DATA_DIR / "train.csv")
    X = build_features(train)
    preds = predict_logit(X)
    true = train["Irrigation_Need"].to_numpy()

    correct_mask = preds == true
    error_mask = ~correct_mask
    n_errors = int(error_mask.sum())
    n_total = len(train)
    error_rate = n_errors / n_total

    _section("Overall error rate")
    print(f"  Total samples : {n_total:,}")
    print(f"  Errors        : {n_errors:,}  ({error_rate:.4%})")

    errors_df = train.filter(pl.Series(error_mask))
    correct_df = train.filter(pl.Series(correct_mask))

    pred_series = pl.Series("pred", preds)
    errors_df = errors_df.with_columns(
        pred_series.filter(pl.Series(error_mask)).alias("pred")
    )

    # ------------------------------------------------------------------ #
    # 1. Which (true, pred) pairs are confused?
    # ------------------------------------------------------------------ #
    _section("Confusion pairs (true → predicted)")
    confusion: dict[tuple[str, str], int] = {}
    for t, p in zip(true[error_mask], preds[error_mask]):
        confusion[(t, p)] = confusion.get((t, p), 0) + 1
    for (t, p), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"    {t:8s} → {p:8s}   {cnt:6,}  ({cnt / n_errors:.1%} of errors)")

    # ------------------------------------------------------------------ #
    # 2. Distance from Deotte thresholds for errors vs correct
    # ------------------------------------------------------------------ #
    _section("Distance from Deotte thresholds  (errors vs correct)")
    for feat, thresh in THRESHOLDS.items():
        err_vals = errors_df[feat].to_numpy()
        cor_vals = correct_df[feat].to_numpy()
        err_dist = np.abs(err_vals - thresh)
        cor_dist = np.abs(cor_vals - thresh)
        print(
            f"  {feat:<30}  "
            f"errors median dist={np.median(err_dist):.2f}  "
            f"correct median dist={np.median(cor_dist):.2f}"
        )

    # ------------------------------------------------------------------ #
    # 3. Numeric feature distributions: errors vs correct
    # ------------------------------------------------------------------ #
    _section("Numeric feature means  (errors vs correct)")
    print(f"  {'Feature':<35} {'Error mean':>12} {'Correct mean':>13}")
    for feat in NUMERIC_FEATURES:
        e_mean = errors_df[feat].mean()
        c_mean = correct_df[feat].mean()
        print(f"  {feat:<35} {e_mean:>12.3f} {c_mean:>13.3f}")

    # ------------------------------------------------------------------ #
    # 4. Categorical feature distributions among errors
    # ------------------------------------------------------------------ #
    _section("Categorical feature distributions  (errors)")
    for feat in CAT_FEATURES:
        print(f"\n  {feat}:")
        _value_counts(errors_df[feat])

    # ------------------------------------------------------------------ #
    # 5. Do unused features cleanly split errors?
    #    For each unused numeric: compare median in errors split by true class
    # ------------------------------------------------------------------ #
    _section("Unused numeric features — error breakdown by true class")
    unused_numeric = [
        "Soil_pH",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Humidity",
        "Sunlight_Hours",
        "Field_Area_hectare",
        "Previous_Irrigation_mm",
    ]
    header = f"  {'Feature':<30}" + "".join(f"  {c:<10}" for c in CLASSES)
    print(header)
    for feat in unused_numeric:
        row_str = f"  {feat:<30}"
        for cls in CLASSES:
            subset = errors_df.filter(pl.col("Irrigation_Need") == cls)[feat]
            med = subset.median() if len(subset) > 0 else float("nan")
            row_str += f"  {med:<10.3f}"
        print(row_str)

    # ------------------------------------------------------------------ #
    # 6. Are errors concentrated near threshold boundaries?
    #    Count errors where feature is within ±X% of threshold
    # ------------------------------------------------------------------ #
    _section("Errors near threshold boundaries (within ±10% of threshold value)")
    for feat, thresh in THRESHOLDS.items():
        margin = thresh * 0.10
        near_mask = (train[feat] - thresh).abs() <= margin
        near_errors = int((pl.Series(error_mask) & near_mask).sum())
        near_total = int(near_mask.sum())
        far_errors = int((pl.Series(error_mask) & ~near_mask).sum())
        far_total = int((~near_mask).sum())
        near_rate = near_errors / near_total if near_total else 0.0
        far_rate = far_errors / far_total if far_total else 0.0
        print(
            f"  {feat:<30}  "
            f"near-threshold error rate={near_rate:.4%} ({near_errors}/{near_total})  "
            f"far error rate={far_rate:.4%} ({far_errors}/{far_total})"
        )

    # ------------------------------------------------------------------ #
    # 7. Save error rows for manual inspection
    # ------------------------------------------------------------------ #
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "formula_errors.csv"
    errors_df.write_csv(out)
    print(f"\nError rows saved → {out}  ({n_errors:,} rows)")


if __name__ == "__main__":
    main()
