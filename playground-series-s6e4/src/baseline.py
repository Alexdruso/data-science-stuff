"""Baseline model for PS S6E4 - Predicting Irrigation Need."""

import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    return train, test


def main() -> None:
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(train.head())


if __name__ == "__main__":
    main()
