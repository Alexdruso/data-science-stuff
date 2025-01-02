from pathlib import Path
import pandas as pd
import polars as pl
from typing import Union
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSeriesKey:
    country: str
    store: str
    product: str

    def to_index(self) -> tuple[str, str, str]:
        return (self.country, self.store, self.product)


def load_data(
    path: Union[str, Path],
    *,
    multi_target: bool,
    train_set: bool,
) -> pd.DataFrame:
    path = Path(path)
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
            (data["country"] + "_" + data["store"] + "_" + data["product"]).alias(
                "identifier"
            )
        )

        # Drop the original columns before pivoting
        data = data.drop(["country", "store", "product"])

        # Pivot the data to merge rows with the same date
        data = data.pivot(
            values=[col for col in data.columns if col not in ["date", "identifier"]],
            index=["date"],
            on="identifier",
            aggregate_function=None,
        )

    return data.to_pandas()


if __name__ == "__main__":
    print(load_data("playground_series_s5e1/data", train_set=True, multi_target=True))
