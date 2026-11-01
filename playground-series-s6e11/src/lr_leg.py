"""Spline logistic regression leg (GAM-like) — the generator is near-additive in logit.

    python src/lr_leg.py [--knots 10] [--C 1.0] [--inter]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import BOOL_COLS, CAT_COLS, NUM_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment, run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
LOW_CARD = ["txn_count_last_24h", "cvv_retry_count", "time_of_day_hour", "day_of_week", "prior_disputes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knots", type=int, default=10)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--inter", action="store_true", help="pairwise interactions of booleans+onehots")
    a = ap.parse_args()
    name = f"lrspl_k{a.knots}{'_inter' if a.inter else ''}"
    train = build_features(pl.read_csv(DATA_DIR / "train.csv")).to_pandas()
    test = build_features(pl.read_csv(DATA_DIR / "test.csv")).to_pandas()
    cont = [c for c in NUM_COLS if c not in LOW_CARD]
    disc = CAT_COLS + LOW_CARD
    cols = cont + disc + BOOL_COLS
    X, X_test, y = train[cols], test[cols], train[TARGET].to_numpy().astype(int)

    def pipe() -> object:
        steps = [("s", SplineTransformer(n_knots=a.knots, degree=3, knots="quantile"), cont),
                 ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False), disc),
                 ("b", "passthrough", BOOL_COLS)]
        ct = ColumnTransformer(steps)
        if a.inter:
            inter = ColumnTransformer(
                [("ib", PolynomialFeatures(2, interaction_only=True, include_bias=False),
                  BOOL_COLS + ["cvv_retry_count", "merchant_risk_score", "velocity_score"])])
            from sklearn.pipeline import FeatureUnion
            from sklearn.preprocessing import StandardScaler
            ct = FeatureUnion([("main", ct), ("inter", make_pipeline(inter, StandardScaler()))])
        return make_pipeline(ct, LogisticRegression(C=a.C, max_iter=3000))

    def fit_fold(x_tr, y_tr, x_va, y_va, x_te, fold):  # type: ignore[no-untyped-def]
        m = pipe().fit(x_tr, y_tr)  # type: ignore[attr-defined]
        va, te = m.predict_proba(x_va)[:, 1], m.predict_proba(x_te)[:, 1]
        print(f"  fold {fold}: {roc_auc_score(y_va, va):.5f}", flush=True)
        return va, te

    res = run_cv(X, y, X_test, fit_fold, score_fn=lambda u, v: float(roc_auc_score(u, v)))
    auc = float(roc_auc_score(y, res.oof))
    print(f"{name}: OOF AUC {auc:.5f}")
    save_predictions(RESULTS_DIR, name, oof=res.oof, test=res.test, train_ids=train["id"].to_numpy(),
                     test_ids=test["id"].to_numpy(), folds=fold_assignment(y))
    save_cv_result(RESULTS_DIR, name, res.fold_scores, auc, metric_name="auc", params=vars(a))
    write_submission(SUBMISSIONS_DIR, f"{name}.csv", test["id"].to_numpy(), TARGET, res.test)


if __name__ == "__main__":
    main()
