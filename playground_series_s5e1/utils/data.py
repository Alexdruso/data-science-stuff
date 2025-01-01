from pathlib import Path
import pandas as pd
import polars as pl

def load_data(
    path: str | Path,
    *,
    multi_target: bool,
    train_set: bool,
) -> pd.DataFrame:
    path: Path = Path(path)
    file_name: str = "train.csv" if train_set else "test.csv"
    path = path / file_name

    data: pl.DataFrame = pl.read_csv(
        source=path,
        try_parse_dates=True,
        schema_overrides={
            "country": pl.Categorical,
            "store": pl.Categorical,
            "product": pl.Categorical,
        },
    )

    if multi_target:
        data = data.with_columns(
            (data["country"] + "_" + data["store"] + "_" + data["product"]).alias("identifier")
        )

        # Drop the original columns before pivoting
        data = data.drop(["country", "store", "product"])

        # Pivot the data to merge rows with the same date
        data = data.pivot(
            values=[col for col in data.columns if col not in ["date", "identifier"]],
            index=["date"],
            columns="identifier",
            aggregate_function=None
        )

    return data.to_pandas()
