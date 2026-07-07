"""LGBM leg with driver-posterior features (plan #2, the FE centerpiece).

Aux LGBMs predict each key driver from the other features, trained
TRANSDUCTIVELY on train+test rows where that driver is observed. The label is
never touched, so this is leak-free; including test rows means the aux models
fit under the union of both mask mechanisms. Their class posteriors are
appended as 9 features -- a soft "probably-high-stress" estimate exactly where
the driver is NaN, i.e. the region holding 86% of the remaining error. Distinct
from the Day-2 marginalization probes: those replaced predictions, this adds
inputs and lets the trees decide.

The label model is baseline.py verbatim (same params, folds, finalize), so at
the same seed/tag surface the delta vs the plain lgbm leg is the feature effect
alone. Run under the repair for surface consistency with today's legs:

  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 \
      ../.venv/bin/python src/train_dp.py           # compare to lgbm_r_s42

Gate (LB-blind protocol): weighted OOF delta vs lgbm_r_s42 > +0.001 solo, or a
blend-level gain at the same margin; below that it is not queued.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))
from baseline import run
from train_common import Dataset, finalize, load_dataset

# driver -> (source column, fixed level order). sleep_bucket discretizes
# sleep_duration at the rule thresholds from the headroom analysis
# (stress=high & sleep<6 -> unhealthy, >=6 -> at-risk; fit needs >=7).
DRIVERS: dict[str, tuple[str, list[str] | None]] = {
    "stress_level": ("stress_level", ["low", "medium", "high"]),
    "activity": ("physical_activity_level", ["sedentary", "moderate", "active"]),
    "sleep_bucket": ("sleep_duration", None),
}
SLEEP_EDGES = [6.0, 7.0]

AUX_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "random_state": 42,
    "verbose": -1,
}


def aux_posteriors(ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one aux model per driver; return posterior frames for train and test."""
    full = pd.concat([ds.train, ds.test], ignore_index=True)
    for col in ds.cat_cols:
        full[col] = full[col].astype("category")
    n_train = len(ds.train)

    tr_cols: dict[str, np.ndarray] = {}
    te_cols: dict[str, np.ndarray] = {}
    for name, (src, levels) in DRIVERS.items():
        observed = full[src].notna().to_numpy()
        if levels is None:
            y_aux = np.digitize(full[src].to_numpy(dtype=float), SLEEP_EDGES)
        else:
            y_aux = (
                full[src].astype(pd.CategoricalDtype(levels)).cat.codes.to_numpy()
            )
        X_aux = full.drop(columns=[src])
        model = LGBMClassifier(**AUX_PARAMS)
        model.fit(X_aux[observed], y_aux[observed])
        assert list(model.classes_) == [0, 1, 2], model.classes_
        proba = model.predict_proba(X_aux)
        acc_obs = float((proba[observed].argmax(1) == y_aux[observed]).mean())
        print(
            f"  aux {name}: trained on {int(observed.sum()):,} observed rows, "
            f"in-sample acc {acc_obs:.4f}"
        )
        for k in range(3):
            tr_cols[f"dp_{name}_{k}"] = proba[:n_train, k]
            te_cols[f"dp_{name}_{k}"] = proba[n_train:, k]
    return (
        pd.DataFrame(tr_cols, index=ds.train.index),
        pd.DataFrame(te_cols, index=ds.test.index),
    )


def main() -> None:
    ds = load_dataset()
    print("Fitting driver-posterior aux models (transductive, label-free):")
    aux_tr, aux_te = aux_posteriors(ds)
    ds.train = pd.concat([ds.train, aux_tr], axis=1)
    ds.test = pd.concat([ds.test, aux_te], axis=1)
    ds.feature_cols = list(ds.train.columns)
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape} (+9 dp features)")

    oof_proba, test_proba, fold_scores = run(ds)
    finalize("lgbm_dp", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
