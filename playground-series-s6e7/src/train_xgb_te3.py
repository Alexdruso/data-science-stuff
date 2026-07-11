"""XGBoost + v0.7 recipe (all-13 string TE + regularized-shallow config) — O1.

Fourth library on the recipe that upgraded HGBC (0.9504), realmlp (0.9505) and
LGBM (0.9503). Config translated from Susol's tuned HGBC: lossguide growth with
33 leaves, min_child_weight ~300 (his min_samples_leaf), colsample 0.82,
max_bin 237, lr 0.0627. Baseline: xgboost_bagw 0.9488. Pass = >= 0.9500.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_xgb_te3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_xgboost import get_device
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

XGB_TE3_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "n_estimators": 1000,
    "learning_rate": 0.0627037115235577,
    "grow_policy": "lossguide",
    "max_leaves": 33,
    "max_depth": 0,
    "min_child_weight": 300,
    "colsample_bytree": 0.820265066682815,
    "max_bin": 237,
    "tree_method": "hist",
    "device": get_device(),
    "enable_categorical": True,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


def main() -> None:
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    print(f"XGB te3: {len(te_names)} TE cols + regularized-shallow config, "
          f"device={XGB_TE3_PARAMS['device']}")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
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
        w_tr = compute_sample_weight("balanced", ds.y[tr_idx])
        model = XGBClassifier(**{**XGB_TE3_PARAMS, "random_state": seed})
        model.fit(
            X_tr,
            ds.y[tr_idx],
            sample_weight=w_tr,
            eval_set=[(X_va, ds.y[val_idx])],
            verbose=False,
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"xgb_te3_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("xgb_te3", ds, oof, test, fold_scores)
    clear_ckpt(f"xgb_te3_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
