"""LGBM canary for the exact-rule deduction features (user idea, 07-10 night).

Adds rule_features (three-valued deduced label set from the KNOWN generation
tree + determined flag) to the baseline LGBM leg. Honest expectation: flat for
trees on complete rows (they learned the rule from data); the live question is
partially-missing rows, where the deduction is noise-free while the empirical
posterior is not.

Run: S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_rule.py
Paired reference: lgbm_r_s42 = 0.9478 weighted (same seed/folds/surface).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from rule_features import add_rule_features
from train_common import N_CLASSES, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

from train_common import SEEDS  # noqa: E402


def main() -> None:
    ds = load_dataset()
    X = add_rule_features(ds.train)
    X_test = add_rule_features(ds.test)
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    det = X["rule_determined"]
    print(
        f"rule_set levels: {X['rule_set'].nunique()}; "
        f"determined rows: {det.mean():.1%} "
        f"(of missing-driver rows: "
        f"{det[ds.train[['sleep_duration', 'stress_level', 'physical_activity_level']].isna().any(axis=1)].mean():.1%})"
    )
    params = load_params()

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"rule_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("lgbm_rule", ds, oof, test, fold_scores)
    clear_ckpt(f"rule_s{SEEDS[0]}")
    _ = np.asarray  # keep numpy import for type checkers


if __name__ == "__main__":
    main()
