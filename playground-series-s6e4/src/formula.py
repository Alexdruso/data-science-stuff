"""Chris Deotte's exact formula for PS S6E4 - Predicting Irrigation Need."""

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

# Logit coefficients: [Low, Medium, High]
INTERCEPTS = np.array([16.3173, 4.6524, -20.9697])
COEFS = np.array([
    # soil_lt_25
    [-11.0237, 0.3290, 10.6947],
    # temp_gt_30
    [-5.8559, -0.0204, 5.8763],
    # rain_lt_300
    [-10.8500, 0.1542, 10.6958],
    # wind_gt_10
    [-5.8284, 0.0841, 5.7444],
    # cgs_Flowering
    [-5.4155, 0.3586, 5.0569],
    # cgs_Harvest
    [5.5073, -0.1348, -5.3725],
    # cgs_Sowing
    [5.2299, -0.3547, -4.8752],
    # cgs_Vegetative
    [-5.4617, 0.3334, 5.1283],
    # mulching_No
    [-3.0014, 0.1883, 2.8131],
    # mulching_Yes
    [2.8613, 0.0142, -2.8755],
])
CLASSES = ["Low", "Medium", "High"]


def build_features(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack([
        (df["Soil_Moisture"] < 25).cast(pl.Int8).to_numpy(),
        (df["Temperature_C"] > 30).cast(pl.Int8).to_numpy(),
        (df["Rainfall_mm"] < 300).cast(pl.Int8).to_numpy(),
        (df["Wind_Speed_kmh"] > 10).cast(pl.Int8).to_numpy(),
        (df["Crop_Growth_Stage"] == "Flowering").cast(pl.Int8).to_numpy(),
        (df["Crop_Growth_Stage"] == "Harvest").cast(pl.Int8).to_numpy(),
        (df["Crop_Growth_Stage"] == "Sowing").cast(pl.Int8).to_numpy(),
        (df["Crop_Growth_Stage"] == "Vegetative").cast(pl.Int8).to_numpy(),
        (df["Mulching_Used"] == "No").cast(pl.Int8).to_numpy(),
        (df["Mulching_Used"] == "Yes").cast(pl.Int8).to_numpy(),
    ])


def predict_logit(X: np.ndarray) -> np.ndarray:
    logits = INTERCEPTS + X @ COEFS
    # stable softmax (not needed for argmax but good practice)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    proba = exp / exp.sum(axis=1, keepdims=True)
    return np.array(CLASSES)[proba.argmax(axis=1)]


def predict_rules(df: pl.DataFrame) -> np.ndarray:
    high_score = (
        2 * (df["Soil_Moisture"] < 25).cast(pl.Int32)
        + 2 * (df["Rainfall_mm"] < 300).cast(pl.Int32)
        + 1 * (df["Temperature_C"] > 30).cast(pl.Int32)
        + 1 * (df["Wind_Speed_kmh"] > 10).cast(pl.Int32)
    ).to_numpy()

    low_score = (
        2 * df["Crop_Growth_Stage"].is_in(["Harvest", "Sowing"]).cast(pl.Int32)
        + 1 * (df["Mulching_Used"] == "Yes").cast(pl.Int32)
    ).to_numpy()

    score = high_score - low_score
    labels = np.where(score <= 0, "Low", np.where(score <= 3, "Medium", "High"))
    return labels


def accuracy(pred: np.ndarray, actual: pl.Series) -> float:
    return float((pred == actual.to_numpy()).mean())


def main() -> None:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Sanity check on train
    X_train = build_features(train)
    logit_train = predict_logit(X_train)
    rules_train = predict_rules(train)
    true_labels = train["Irrigation_Need"]
    print(f"Train logit accuracy:  {accuracy(logit_train, true_labels):.4f}")
    print(f"Train rules accuracy:  {accuracy(rules_train, true_labels):.4f}")

    # Generate test predictions
    X_test = build_features(test)
    test_ids = test["id"].to_numpy()

    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    import pandas as pd

    logit_preds = predict_logit(X_test)
    out_logit = SUBMISSIONS_DIR / "formula_logit.csv"
    pd.DataFrame({"id": test_ids, "Irrigation_Need": logit_preds}).to_csv(out_logit, index=False)
    print(f"Logit submission → {out_logit}")

    rules_preds = predict_rules(test)
    out_rules = SUBMISSIONS_DIR / "formula_rules.csv"
    pd.DataFrame({"id": test_ids, "Irrigation_Need": rules_preds}).to_csv(out_rules, index=False)
    print(f"Rules submission → {out_rules}")

    # Class distribution in test predictions
    for name, preds in [("logit", logit_preds), ("rules", rules_preds)]:
        unique, counts = np.unique(preds, return_counts=True)
        dist = {k: v for k, v in zip(unique, counts)}
        print(f"Test {name} dist: {dist}")


if __name__ == "__main__":
    main()
