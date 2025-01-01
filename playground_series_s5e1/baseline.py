import pandas as pd
import argparse
from pycaret.time_series import setup

def main(data_path: str) -> None:
    train_set: pd.DataFrame = pd.read_csv(source=f"{data_path}/train.csv")
    print(train_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str)
    args = parser.parse_args()
    main(getattr(args, "data", ""))
