"""LightGBM leg with tuned parameters — the route from Optuna to an OOF array.

    python src/train_lgbm.py --arm s1 --seeds 42,43,44

`tune_lgbm.py` writes `results/best_params.json`, but `run_arm.py::LGBM_PARAMS`
is hard-coded and never calls `load_params`, and `baseline.py` (which does) is
frozen on the `s0` surface. So without this script a tuned parameter set has no
way to become a 5-fold `s1` OOF array.

The tag is `lgbmt_<arm>_s<seed>`, so `ensemble.resolve_arm` still reads the
surface off the middle segment and the tuned leg blends with the untuned one
without ever being mistaken for an arm-delta read.

Device is locked to CPU like the rest of the sweep: the cuda/cpu gap on this box
is 0.0003 AUC, the same size as the effect this leg is being compared for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from legs import run_leg
from run_arm import LGBM_PARAMS, ARMS

from data_science_stuff.kaggle.io import competition_dirs, load_params

_, RESULTS_DIR, _ = competition_dirs(__file__)


def make_fit(seed: int, params: dict[str, object]):  # noqa: ANN201 - closure over the seed
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    return fit_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--params", default="best_params.json")
    args = parser.parse_args()

    params = dict(load_params(RESULTS_DIR, LGBM_PARAMS, args.params))
    params["device_type"] = "cpu"  # locked — see the module docstring
    if not (RESULTS_DIR / args.params).exists():
        print(f"warning: no {args.params} — this is the UNTUNED leg, identical to run_arm")
    print(f"lgbmt on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    print(f"params: {params}", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        run_leg("lgbmt", args.arm, seed, make_fit(seed, params))


if __name__ == "__main__":
    main()
