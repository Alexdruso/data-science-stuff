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

from data_science_stuff.kaggle.io import competition_dirs, load_params

_, RESULTS_DIR, _ = competition_dirs(__file__)

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
    # 10, not 12: this leg is run concurrently with the GPU CatBoost leg, which
    # still needs CPU for data prep. Threads only affect wall clock, not the
    # fitted model, so this does not touch any paired comparison.
    "n_jobs": 10,
    "verbosity": 0,
}


def make_fit(seed: int, params: dict[str, object]):  # noqa: ANN201 - closure
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = XGBClassifier(**{**params, "random_state": seed})
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
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="load results/best_params_xgboost.json instead of the hand-set PARAMS, "
        "and tag the arrays `xgbt_` so tuned and untuned legs never collide",
    )
    args = parser.parse_args()

    # A distinct tag matters more than it looks: the tuning bar is a *paired*
    # tuned-vs-untuned read, so both sets of arrays have to coexist. Reusing the
    # `xgb_` tag would overwrite the very baseline the comparison needs.
    params, name = dict(PARAMS), "xgb"
    if args.tuned:
        params = dict(load_params(RESULTS_DIR, PARAMS, "best_params_xgboost.json"))
        params.setdefault("early_stopping_rounds", 100)
        # The tuning study wrote n_jobs=12 into the file. Pin it back to the
        # leg's own value so the paired read runs at the same thread count as
        # the untuned reference and leaves cores for a concurrent GPU leg.
        # Threads are wall clock only (see PARAMS above), so this is not a
        # change to either arm of the comparison.
        params["n_jobs"] = PARAMS["n_jobs"]
        name = "xgbt"
    print(f"{name} on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        run_leg(name, args.arm, seed, make_fit(seed, params))


if __name__ == "__main__":
    main()
