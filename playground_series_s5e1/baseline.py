import polars as pl
import argparse

def main(data_path: str) -> None:
    train_set: pl.DataFrame = pl.read_csv(source=f"{data_path}/train.csv")

    print(train_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str)
    args = parser.parse_args()
    main(getattr(args, "data", ""))
