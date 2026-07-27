"""LGBM + v0.7 recipe (all-13 string TE + regularized-shallow config) — E2.

The hgbc_te3d ablation showed the tuned regularized-shallow config carries
~+0.0007 of the v0.7 gain on HGBC. This tests whether the config x TE pairing
transfers to a second library: LGBM with Susol's HGBC hyperparameters
translated (num_leaves 33, min_child_samples 298, feature_fraction 0.82,
learning_rate 0.0627, max_bin 237) + the same 39-column string TE block.
Baselines: lgbm_bagw 0.9493, lgbm_te_u (default config, numerics-only TE)
0.9491. Pass = >= 0.9500.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_lgbm_te3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from lgbm_device import get_lgbm_device
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

_DEVICE_TYPE, _N_JOBS = get_lgbm_device()
LGBM_TE3_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "n_estimators": 1000,
    "learning_rate": 0.0627037115235577,
    "num_leaves": 33,
    "min_child_samples": 298,
    "feature_fraction": 0.820265066682815,
    "max_bin": 237,
    "class_weight": "balanced",
    "verbose": -1,
    "n_jobs": _N_JOBS,
    "device_type": _DEVICE_TYPE,
}


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None, "S6E7_REPAIR_FIT not wired for this trainer"
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    print(f"LGBM te3: {len(te_names)} TE cols + regularized-shallow config")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        # seeded inner CV (Day-10 audit finding, ported from zoo_common's
        # te13_block_for_fold: sklearn 1.9's cv=<int> default is an UNSEEDED
        # shuffled StratifiedKFold — nondeterministic TE draws on every rerun).
        enc = TargetEncoder(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            smooth="auto",
            target_type="multiclass",
        )
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = LGBMClassifier(**{**LGBM_TE3_PARAMS, "random_state": seed})
        model.fit(
            X_tr,
            ds.y[tr_idx],
            eval_set=[(X_va, ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"lgbm_te3_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("lgbm_te3", ds, oof, test, fold_scores)
    clear_ckpt(f"lgbm_te3_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
