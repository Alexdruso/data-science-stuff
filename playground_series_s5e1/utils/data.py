from pathlib import Path
import pandas as pd
import polars as pl


def load_data(
    path: str | Path,
    *,
    train_set: bool = True,
) -> pd.DataFrame:
    path: Path = Path(path)
    file_name: str = "train.csv" if train_set else "test.csv"
    path = path / file_name

    data: pl.DataFrame = pl.read_csv(
        file_name,
        try_parse_dates=True,
        schema_overrides={
            "country": pl.Categorical,
            "store": pl.Categorical,
            "product": pl.Categorical,
        },
    )

    return data.to_pandas()
