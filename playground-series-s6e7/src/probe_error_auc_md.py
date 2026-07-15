"""Day-13 PM — missing-driver residual scan (forensics idea #1, the scope gap).

The Day-10 archaeologist kill-shot (P(flip | triple) independent of everything)
ran on COMPLETE-driver rows only; the missing-driver region — 26% of rows, 86%
of residual error — had its saturation asserted from Day-2-era probes (pre-te13
frontier). This closes the gap at the modern frontier: are lrstack_u4's errors
on missing-driver rows predictable from features beyond the core's own
confidence? Unlike probe_error_auc (global fit, subset read), the predictor
here is FIT on the subset so in-region structure can't be swamped globally.

  conf — u4 per-class probs, max, margin, entropy
  feat — raw features (cats categorical, NaN native)
  both — features + confidence

Headroom evidence = AUC(both) − AUC(conf) on the subset. <0.02 = the region's
saturation is finally proven, not asserted, at the deployed frontier.

Run: python src/probe_error_auc_md.py -> results/probe_error_auc_md.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import weighted_predict

CORE = "lrstack_u4_r"
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]


def main() -> None:
    ds = load_dataset()
    core = np.load(RESULTS_DIR / f"oof_{CORE}.npy")
    dwj = json.load(open(RESULTS_DIR / f"decision_weights_{CORE}.json"))
    w = (
        np.array(list(dwj["decision"].values()))
        if "decision" in dwj
        else np.array([dwj[c] for c in CLASSES])
    )
    pred = weighted_predict(core, w)
    err = (pred != ds.y).astype(np.int64)

    md = ds.train[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    print(
        f"missing-driver rows: {md.sum():,} ({md.mean():.1%}); "
        f"core error rate there {err[md].mean():.4f} vs complete {err[~md].mean():.4f}"
    )

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
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
    y_sub = err[md]
    variants = {
        "conf": conf[md],
        "feat": X[md],
        "both": pd.concat([X, conf], axis=1)[md],
    }
    params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    lines = [
        f"missing-driver residual scan on {CORE}: n={md.sum():,}, "
        f"err-rate {y_sub.mean():.4f}"
    ]
    aucs = {}
    for name, Xv in variants.items():
        oof_p = np.zeros(len(y_sub))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in skf.split(Xv, y_sub):
            m = LGBMClassifier(**params)
            m.fit(Xv.iloc[tr_idx], y_sub[tr_idx])
            oof_p[val_idx] = m.predict_proba(Xv.iloc[val_idx])[:, 1]
        aucs[name] = roc_auc_score(y_sub, oof_p)
        lines.append(f"  {name:<5} AUC {aucs[name]:.4f}")
    inc = aucs["both"] - aucs["conf"]
    lines.append(f"increment AUC(both)-AUC(conf) = {inc:+.4f}")
    lines.append(
        "verdict: " + (
            "in-region feature structure beyond confidence — A LEAD"
            if inc > 0.02
            else "missing-driver saturation PROVEN at the deployed frontier"
        )
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_error_auc_md.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
