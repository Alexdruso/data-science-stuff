"""XGBoost leg — diversity against the LightGBM leg for the blend.

    python src/train_xgboost.py --arm s1 --seeds 42

Categoricals go in via pandas `category` dtype with `enable_categorical=True`;
`build_matrices` has already unioned the level sets across train and test so the
codes match. NaN is handled natively, same as LightGBM, so the arm's transform
(if any) is the only thing that changes null handling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from legs import run_leg
from run_arm import ARMS

PARAMS: dict[str, object] = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "enable_categorical": True,
    "tree_method": "hist",
    "n_jobs": 12,
    "verbosity": 0,
}


def make_fit(seed: int):  # noqa: ANN201 - closure over the seed
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = XGBClassifier(**PARAMS, random_state=seed)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        return (
            model.predict_proba(x_va)[:, 1],
            model.predict_proba(x_te)[:, 1],
        )

    return fit_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    print(f"xgb on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        run_leg("xgb", args.arm, seed, make_fit(seed))


if __name__ == "__main__":
    main()
