"""CatBoost leg — ordered boosting gives genuine diversity against LGBM/XGB.

    python src/train_catboost.py --arm s1 --seeds 42
    S6E8_CATBOOST_DEVICE=CPU python src/train_catboost.py --arm s1

GPU settings on this 6GB RTX 2060 follow the hard-won recipe: `max_ctr_complexity=1`,
no `eval_set`, and a capped `gpu_ram_part`. Without those CatBoost silently
falls back to CPU and the run takes hours. Dropping `eval_set` means no early
stopping, so the iteration count is fixed rather than searched.

CatBoost rejects NaN in categorical features, so the three (uninformative)
string columns are filled with an explicit `__NA__` level — which is the honest
encoding anyway, since a missing category here is its own level.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_CATEGORICAL_COLS
from legs import run_leg
from run_arm import ARMS

DEVICE = os.environ.get("S6E8_CATBOOST_DEVICE", "GPU").upper()

PARAMS: dict[str, object] = {
    "iterations": 2000,
    "learning_rate": 0.06,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "max_ctr_complexity": 1,
    "verbose": 0,
    "allow_writing_files": False,
}


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in RAW_CATEGORICAL_COLS:
        out[col] = out[col].astype(object).fillna("__NA__").astype(str)
    return out


def make_fit(seed: int):  # noqa: ANN201 - closure over the seed
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        _y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_tr, x_va, x_te = _prepare(x_tr), _prepare(x_va), _prepare(x_te)
        params = dict(PARAMS, random_seed=seed, task_type=DEVICE)
        if DEVICE == "GPU":
            params["gpu_ram_part"] = 0.8
        model = CatBoostClassifier(**params)
        # No eval_set on purpose — see the module docstring.
        model.fit(x_tr, y_tr, cat_features=RAW_CATEGORICAL_COLS)
        val = model.predict_proba(x_va)[:, 1]
        test = model.predict_proba(x_te)[:, 1]
        del model
        return val, test

    return fit_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    print(f"catboost ({DEVICE}) on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        run_leg("cat", args.arm, seed, make_fit(seed))


if __name__ == "__main__":
    main()
