import pandas as pd
import argparse
from pycaret.time_series import setup, compare_models


def main(data_path: str) -> None:
    train_set: pd.DataFrame = pd.read_csv(filepath_or_buffer=f"{data_path}/train.csv")
    train_set["date"] = pd.to_datetime(train_set["date"])
    print(train_set)
    print(train_set.dtypes)
    train_set = train_set.dropna(subset=["num_sold"])

    experiment = setup(
        data=train_set, target="num_sold", index="date", ignore_features=["id"]
    )

    best_model = compare_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str)
    args = parser.parse_args()
    main(getattr(args, "data", ""))
