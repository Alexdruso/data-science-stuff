"""PR-25: honest transductive pseudo-labelling of the test rows (LGBM extra-trees leg).

    python src/pseudo_lgbm.py --weights 1.0 0.3

Per fold k (run_cv's splitter):
  stage 1 — the leg as usual (route TE fit on the fold's training split, original
            rows appended, og/agg/auxall extras) → val AUC (the paired control) and
            test probabilities p_k from a model that never saw fold k;
  stage 2 — refit on the same training rows + every test row with SOFT label p_k
            (LightGBM `cross_entropy` objective takes labels in [0, 1]), test rows
            weighted w → val AUC.
The val fold never enters either stage's training data and p_k depends only on
the fold's training labels, so stage 2's OOF is as honest as stage 1's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from matrices import build_matrices
from route import add_extras, add_route_global, route_te
from train_lgbm_orig import PARAMS as CONTROL_PARAMS

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.3])
    p.add_argument("--n-jobs", type=int, default=12)
    a = p.parse_args()

    tuned = json.loads((RESULTS_DIR / "best_params_lgbm_xt.json").read_text())
    params = {**CONTROL_PARAMS, **tuned, "n_jobs": a.n_jobs}
    m = build_matrices(with_original=True)
    n_test = len(m.X_test)
    m.X_test = pd.concat([m.X_test, m.X_orig], ignore_index=True)
    add_route_global(m)
    x_orig = m.X_test.iloc[n_test:].reset_index(drop=True)
    m.X_test = m.X_test.iloc[:n_test].reset_index(drop=True)
    add_extras(m, x_orig, ["og", "agg", "auxall"])
    assert m.y_orig is not None

    names = ["stage1", *[f"pl_w{w:g}" for w in a.weights]]
    oof = {n: np.zeros(len(m.y)) for n in names}
    test = {n: np.zeros(n_test) for n in names}
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
    for k, (tr, va) in enumerate(splits, 1):
        t0 = time.time()
        x_tr, x_va, x_te, x_o = route_te(m.X.iloc[tr], m.y[tr], [m.X.iloc[va], m.X_test, x_orig])
        x_fit = pd.concat([x_tr, x_o[x_tr.columns]], ignore_index=True)
        y_fit = np.concatenate([m.y[tr], m.y_orig]).astype(float)
        y_va = m.y[va]
        s1 = LGBMClassifier(**params)
        s1.fit(x_fit, y_fit.astype(int), eval_set=[(x_va, y_va)],
               callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        oof["stage1"][va] = s1.predict_proba(x_va)[:, 1]
        p_te = s1.predict_proba(x_te)[:, 1]
        test["stage1"] += p_te / 5
        line = [f"stage1 {roc_auc_score(y_va, oof['stage1'][va]):.5f} (it {s1.best_iteration_})"]
        for w, name in zip(a.weights, names[1:]):
            x2 = pd.concat([x_fit, x_te[x_fit.columns]], ignore_index=True)
            y2 = np.concatenate([y_fit, p_te])
            sw = np.concatenate([np.ones(len(y_fit)), np.full(len(p_te), w)])
            reg_params = {k2: v for k2, v in params.items() if k2 not in ("objective", "is_unbalance")}
            s2 = LGBMRegressor(**{**reg_params, "objective": "cross_entropy"})
            s2.fit(x2, y2, sample_weight=sw, eval_set=[(x_va, y_va.astype(float))],
                   eval_metric="auc",
                   callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
            oof[name][va] = s2.predict(x_va)
            test[name] += s2.predict(x_te) / 5
            line.append(f"{name} {roc_auc_score(y_va, oof[name][va]):.5f} (it {s2.best_iteration_})")
        print(f"fold {k}: " + "  ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)

    base = roc_auc_score(m.y, oof["stage1"])
    for name in names:
        auc = float(roc_auc_score(m.y, oof[name]))
        run = f"lgbm_xt_og_agg_auxall_orig_{name}"
        np.save(RESULTS_DIR / f"oof_{run}.npy", oof[name])
        np.save(RESULTS_DIR / f"test_{run}.npy", test[name])
        print(f"{run}: OOF {auc:.6f}  delta vs stage1 {auc - base:+.6f}")


if __name__ == "__main__":
    main()
