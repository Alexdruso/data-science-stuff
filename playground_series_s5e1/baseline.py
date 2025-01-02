import pandas as pd
import argparse
from pycaret.time_series import setup, compare_models
from utils.data import load_data


def main(data_path: str) -> None:
    train_set: pd.DataFrame = load_data(
        path=data_path, multi_target=False, train_set=True
    )

    countries: set[str] = set(train_set["country"])
    stores: set[str] = set(train_set["store"])
    products: set[str] = set(train_set["product"])

    train_set = train_set.set_index(keys=["country", "store", "product"])

    print(f"{train_set}")

    for country in countries:
        for store in stores:
            for product in products:
                print(f"Learning to predict: {country=} {store=} {product=}")

                experiment = setup(
                    data=train_set.loc[(country, store, product)],
                    target="num_sold",
                    index="date",
                    ignore_features=["id"],
                )

                best_model = compare_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str)
    args = parser.parse_args()
    main(getattr(args, "data", ""))
