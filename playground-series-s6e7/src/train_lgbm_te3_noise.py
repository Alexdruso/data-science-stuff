"""Provable-noise curation arms on the lgbm_te3 recipe (Day-10, TE-eng #1).

~4,800 train rows are rule-DETERMINED by their observed drivers yet carry a
contradicting label — known generator noise (0.78% on determined rows, Day-9).
Mislabeled minority rows carry ~15x metric weight, and the noise poisons the
exact-value TE tables that just paid (v0.7). Four paired seed-42 arms vs
lgbm_te3_u_s42 (0.9503), selected by S6E7_NOISE_ARM:

  nrel  — relabel contradicted rows to the rule label (TE fit + LGBM fit)
  ndrop — drop contradicted rows from training folds entirely
  nsoft — soft label: contradicted rows keep the observed label at weight
          0.008 and gain a rule-labeled copy at weight 0.992 (exact soft-CE);
          TE fit uses the relabeled target (0.992-dominant component)
  ntex  — labels untouched; contradicted rows only EXCLUDED from TE fitting
          (isolates the TE-contamination channel; their own TE values come
          from the full-fold transform)

Val rows and the metric are NEVER touched — OOF is always scored on the
observed labels, so deltas are read against the paired baseline, not an
easier surface.

Run (one arm at a time, unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 S6E7_NOISE_ARM=nrel \
      python src/train_lgbm_te3_noise.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES
from lgbm_device import get_lgbm_device
from rule_features import rule_label_set
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_lgbm_te3 import LGBM_TE3_PARAMS
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

ARM = os.environ.get("S6E7_NOISE_ARM", "nrel")
SOFT_W = 0.992

_DEVICE_TYPE, _N_JOBS = get_lgbm_device()


def main() -> None:
    assert ARM in {"nrel", "ndrop", "nsoft", "ntex"}, ARM
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    rs = rule_label_set(ds.train)
    determined = (~rs.str.contains(r"\|")).to_numpy()
    rule_y = rs.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=float)
    contradicted = determined & (np.nan_to_num(rule_y, nan=-1) != ds.y)
    y_rel = ds.y.copy()
    y_rel[contradicted] = rule_y[contradicted].astype(np.int64)
    print(
        f"arm={ARM}: {int(contradicted.sum())} rule-contradicted rows "
        f"({contradicted.mean():.4%} of train; "
        f"{contradicted.sum() / determined.sum():.4%} of determined)"
    )

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        contr_tr = contradicted[tr_idx]
        clean_tr = tr_idx[~contr_tr]
        dirty_tr = tr_idx[contr_tr]

        # --- TE tables ---------------------------------------------------
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        if ARM == "ndrop" or ARM == "ntex":
            te_fit = enc.fit_transform(te_tr_view.iloc[clean_tr], ds.y[clean_tr])
        else:  # nrel / nsoft — relabeled target
            te_fit = enc.fit_transform(te_tr_view.iloc[tr_idx], y_rel[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        # --- training design per arm -------------------------------------
        if ARM == "nrel":
            fit_idx, y_fit, w_fit = tr_idx, y_rel[tr_idx], None
            X_tr = assemble(X.iloc[fit_idx], te_fit)
        elif ARM == "ndrop":
            fit_idx, y_fit, w_fit = clean_tr, ds.y[clean_tr], None
            X_tr = assemble(X.iloc[fit_idx], te_fit)
        elif ARM == "ntex":
            # clean rows keep cross-fitted TE; dirty rows get full-fold TE
            fit_idx, y_fit, w_fit = tr_idx, ds.y[tr_idx], None
            blk = np.empty((len(tr_idx), len(te_names)))
            blk[~contr_tr] = te_fit
            blk[contr_tr] = enc.transform(te_tr_view.iloc[dirty_tr])
            X_tr = assemble(X.iloc[fit_idx], blk)
        else:  # nsoft — observed label at 1-SOFT_W + rule-labeled copy at SOFT_W
            fit_idx = np.concatenate([tr_idx, dirty_tr])
            y_fit = np.concatenate([ds.y[tr_idx], y_rel[dirty_tr]])
            w_fit = np.ones(len(fit_idx))
            w_fit[: len(tr_idx)][contr_tr] = 1.0 - SOFT_W
            w_fit[len(tr_idx) :] = SOFT_W
            blk = np.concatenate([te_fit, te_fit[contr_tr]])
            X_tr = assemble(X.iloc[fit_idx], blk)

        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = LGBMClassifier(**{**LGBM_TE3_PARAMS, "random_state": seed})
        model.fit(
            X_tr,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_va, ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"lgbm_te3_{ARM}_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize(f"lgbm_te3_{ARM}", ds, oof, test, fold_scores)
    clear_ckpt(f"lgbm_te3_{ARM}_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
