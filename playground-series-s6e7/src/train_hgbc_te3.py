"""Exact Susol v0.7 replica (repo notebook v0.7-hgbc-te.ipynb) — third TE attempt.

The two prior attempts missed his actual recipe. From the notebook source:
  1. TE is fit on ALL 13 RAW columns cast to STRINGS (numerics too — exact
     values; NaN becomes the 'nan'/'missing' string = its own TE level), via
     sklearn TargetEncoder(cv=5, smooth='auto', target_type='multiclass')
     -> 39 TE columns. Our attempts encoded the 7 numerics only.
  2. HGBC uses HIS TUNED config (lr 0.0627, max_iter 300, max_leaf_nodes 33,
     min_samples_leaf 298, l2 0.0289, max_bins 237, max_features 0.82,
     default-loss early stopping) — not our balanced-acc-scored default.
  3. Base features: raw numerics (NaN native) + cats as pandas Categorical
     with 'missing' filled level.

His anchor: OOF 0.95026 (his fold split), LB 0.95036 — no haircut.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3.py

S6E7_NO_CW=1 (Day-10 PM): drop class_weight="balanced" and let the decision
layer do ALL prior correction — the field warns balanced training + tuned
decision weights double-correct inside the sigmoid boundary band (the
15x-upweighted minority side of a 20-40%-flip cell distorts what the tree
estimates there). Artifact name becomes hgbc_te3ncw.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

HGBC_CONFIG: dict[str, object] = {
    "learning_rate": 0.0627037115235577,
    "max_iter": 300,
    "max_leaf_nodes": 33,
    "min_samples_leaf": 298,
    "l2_regularization": 0.028912644384523085,
    "max_bins": 237,
    "max_features": 0.820265066682815,
    "early_stopping": True,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
}


NO_CW = os.environ.get("S6E7_NO_CW") == "1"


def main() -> None:
    config = dict(HGBC_CONFIG)
    name = "hgbc_te3"
    if NO_CW:
        config["class_weight"] = None
        name = "hgbc_te3ncw"
        print("S6E7_NO_CW=1: unweighted training, decision layer corrects priors")
    # S6E7_HGBC_PARAMS (Day-12 tuning): JSON dict merged over HGBC_CONFIG for
    # confirmation runs. The model name does NOT change, so an explicit fresh
    # S6E7_RUN_TAG is required — never overwrite the finals-lineage arrays.
    params_env = os.environ.get("S6E7_HGBC_PARAMS")
    if params_env:
        if not os.environ.get("S6E7_RUN_TAG"):
            raise SystemExit("S6E7_HGBC_PARAMS requires an explicit S6E7_RUN_TAG")
        config.update(json.loads(params_env))
        print(f"S6E7_HGBC_PARAMS override: {json.loads(params_env)}")
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    # TE view: ALL columns as strings (numeric NaN -> 'nan' = its own level)
    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    # S6E7_REPAIR_FIT=1 (Day-13): fit encoder+model on the remasked matrix,
    # predict val/test on the plain one (surface-consistent OOF for the joint
    # u+r stack). Vocab note: remask only converts values to NaN, so the plain
    # val rows' values are covered by the fit vocabulary (unseen -> prior).
    X_fit, te_fit_view = X, te_tr_view
    if ds.train_fit is not None:
        X_fit = ds.train_fit.copy()
        for col in ds.cat_cols:
            X_fit[col] = X_fit[col].fillna("missing").astype(str)
            X_fit[col] = pd.Categorical(X_fit[col], categories=X[col].cat.categories)
        te_fit_view = X_fit[raw_cols].astype(str)
        print("repair-at-fit-only: model/TE fit on remasked rows, val/test plain")
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    print(f"v0.7 replica: TE over {len(raw_cols)} string-cast cols "
          f"({len(te_names)} TE features), tuned HGBC config")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        # seeded inner CV (Day-10 audit finding, ported from zoo_common's
        # te13_block_for_fold: sklearn 1.9's cv=<int> default is an UNSEEDED
        # shuffled StratifiedKFold — nondeterministic TE draws on every rerun).
        enc = TargetEncoder(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            smooth="auto",
            target_type="multiclass",
        )
        te_tr = enc.fit_transform(te_fit_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [
                    base.reset_index(drop=True),
                    pd.DataFrame(blk, columns=te_names),
                ],
                axis=1,
            )

        X_tr = assemble(X_fit.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = HistGradientBoostingClassifier(**config, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    # ckpt tag must carry the surface (Day-13 audit): an rt run crashing without
    # an explicit RUN_TAG must never leave fold ckpts under a u-named dir.
    surface = "rt_u" if ds.train_fit is not None else "u"
    tag = os.environ.get("S6E7_RUN_TAG", f"_{surface}_s{SEEDS[0]}")
    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"{name}{tag}", seed=SEEDS[0]
    )
    finalize(name, ds, oof, test, fold_scores)
    clear_ckpt(f"{name}{tag}")


if __name__ == "__main__":
    main()
