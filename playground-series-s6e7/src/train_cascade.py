"""Chained binary cascade (Day-9 #1, user-flagged 07-10 — last assertion-kill).

s6e6's chain_cascade earned seats in its winning LR stack. Here "hierarchical
2-stage" was Tier-C-killed by "fit<->unhealthy already solved" — the WRONG
stage. Stage 1 = at-risk-vs-minority as a dedicated BINARY problem attacks the
one live boundary (79% of consequential flips) at TRAINING level: a binary
objective changes what the trees LEARN, which the cost-matrix kill (decision
layer only) does not cover — same reason class weights != decision weights.

Stage 1: LGBM binary P(at-risk | x), class-weighted.
Stage 2: LGBM binary P(fit | minority, x) on minority rows only, class-weighted.
Recombine: kaggle.decision.cascade_combine([p_ar, p_fit|min], [0, 1, 2]) ->
P(at-risk)=p_ar, P(fit)=(1-p_ar)p_f, P(unhealthy)=(1-p_ar)(1-p_f).

Run (zoo protocol, m100 repaired surface, ~paired with lgbm_r_s42 0.9478):
  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_cascade.py
Verdict: diag_mlp_transfer signature + seat in the LR-combiner tournament,
NOT solo OOF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from train_common import SEEDS, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

from data_science_stuff.kaggle.decision import cascade_combine


def binary_params() -> dict[str, object]:
    """Baseline (tuned) LGBM params adapted to a binary stage."""
    p = load_params()
    p.pop("num_class", None)
    p["objective"] = "binary"
    p["metric"] = "binary_logloss"
    return p


def main() -> None:
    ds = load_dataset()
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    params = binary_params()

    def fit_binary(tr_X, tr_y, val_X, val_y, seed):  # noqa: ANN001, ANN202
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            tr_X,
            tr_y,
            eval_set=[(val_X, val_y)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == [0, 1], model.classes_
        return model

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = ds.y[tr_idx], ds.y[val_idx]

        # stage 1: at-risk (class 0) vs minority — the live boundary as the objective
        m1 = fit_binary(X_tr, (y_tr == 0).astype(int), X_va, (y_va == 0).astype(int), seed)
        p1_val = m1.predict_proba(X_va)[:, 1]
        p1_test = m1.predict_proba(X_test)[:, 1]

        # stage 2: fit (1) vs unhealthy (2), minority rows only
        min_tr, min_va = y_tr != 0, y_va != 0
        m2 = fit_binary(
            X_tr[min_tr],
            (y_tr[min_tr] == 1).astype(int),
            X_va[min_va],
            (y_va[min_va] == 1).astype(int),
            seed,
        )
        p2_val = m2.predict_proba(X_va)[:, 1]
        p2_test = m2.predict_proba(X_test)[:, 1]

        val_proba = cascade_combine([p1_val, p2_val], [0, 1, 2])
        test_proba = cascade_combine([p1_test, p2_test], [0, 1, 2])
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"cascade_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("cascade", ds, oof, test, fold_scores)
    clear_ckpt(f"cascade_s{SEEDS[0]}")
    _ = np.asarray  # keep numpy import for type checkers


if __name__ == "__main__":
    main()
