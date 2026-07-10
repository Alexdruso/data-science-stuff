"""Ceiling falsifier: are the deployed core's OOF errors predictable? (Day-2 item 4.)

If no model can find structure in WHERE the core errs — beyond what the core's own
confidence already says — the residual error is row-level noise/deleted information
and today's zoo is a hedge-diversity play, not an accuracy play. If errors ARE
predictable from features beyond confidence, that predictive structure is a lead.

Three 5-fold LGBMs predict err = (deployed core prediction != y) on the repaired
surface (the surface the core lives on):
  conf  — core-confidence features only (per-class probs, max, margin, entropy)
  feat  — raw feature matrix only
  both  — features + confidence
Headroom evidence = AUC(both) - AUC(conf). Reported overall and within the
missing-driver / complete-driver regions.

Run: S6E7_REPAIR=1 python src/probe_error_auc.py | tee results/probe_error_auc.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    RESULTS_DIR,
    Dataset,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle_utils import weighted_predict

CORE = "oof_ensemble_r_breadth.npy"
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]


def main() -> None:
    ds: Dataset = load_dataset()
    core = np.load(RESULTS_DIR / CORE)
    assert core.shape == (len(ds.y), 3), core.shape

    # Deployed prediction = robust decision weights on the (precorrected) core OOF.
    w = robust_decision_weights(ds.y, core)
    pred = weighted_predict(core, w)
    err = (pred != ds.y).astype(np.int64)
    print(f"core error rate: {err.mean():.4f} ({err.sum():,} rows)")

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    missing_driver = ds.train[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    print(f"missing-driver rows: {missing_driver.mean():.1%}")

    conf = pd.DataFrame(
        {
            "p0": core[:, 0],
            "p1": core[:, 1],
            "p2": core[:, 2],
            "pmax": core.max(axis=1),
            "margin": np.sort(core, axis=1)[:, -1] - np.sort(core, axis=1)[:, -2],
            "entropy": -(core * np.log(core + 1e-12)).sum(axis=1),
        },
        index=X.index,
    )

    variants: dict[str, pd.DataFrame] = {
        "conf": conf,
        "feat": X,
        "both": pd.concat([X, conf], axis=1),
    }
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    print(f"\n{'model':<6} {'AUC all':>8} {'AUC miss-drv':>13} {'AUC complete':>13}")
    aucs: dict[str, float] = {}
    for name, Xv in variants.items():
        oof_p = np.zeros(len(err))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in skf.split(Xv, err):
            m = LGBMClassifier(**params)
            m.fit(Xv.iloc[tr_idx], err[tr_idx])
            oof_p[val_idx] = m.predict_proba(Xv.iloc[val_idx])[:, 1]
        a_all = roc_auc_score(err, oof_p)
        a_md = roc_auc_score(err[missing_driver], oof_p[missing_driver])
        a_cp = roc_auc_score(err[~missing_driver], oof_p[~missing_driver])
        aucs[name] = a_all
        print(f"{name:<6} {a_all:>8.4f} {a_md:>13.4f} {a_cp:>13.4f}")

    inc = aucs["both"] - aucs["conf"]
    print(f"\nincrement AUC(both) - AUC(conf) = {inc:+.4f}")
    verdict = (
        "errors carry feature structure beyond confidence — a lead"
        if inc > 0.02
        else "row-level limit corroborated: features add ~nothing over confidence"
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
