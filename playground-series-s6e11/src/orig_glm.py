"""GLM fit on the 20k ORIGINAL rows only (the label formula), scored on train/test. No train labels -> no CV.
    python src/orig_glm.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import BOOL_COLS, CAT_COLS, NUM_COLS, TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
tr = build_features(pl.read_csv(DATA_DIR / "train.csv")).to_pandas()
te = build_features(pl.read_csv(DATA_DIR / "test.csv")).to_pandas()
orig = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv").to_pandas()
for d in (tr, te, orig):
    for c in BOOL_COLS:
        d[c] = d[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
y_or = orig[TARGET].astype(int).to_numpy()
for name, num in [("lin", StandardScaler()), ("spl", SplineTransformer(n_knots=5, degree=3))]:
    for C in (0.1, 1.0):
        ct = ColumnTransformer([("n", num, NUM_COLS), ("c", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                                ("b", "passthrough", BOOL_COLS)])
        pipe = make_pipeline(ct, LogisticRegression(C=C, max_iter=5000))
        cv = cross_val_score(pipe, orig, y_or, cv=5, scoring="roc_auc").mean()
        pipe.fit(orig, y_or)
        o, t = pipe.decision_function(tr), pipe.decision_function(te)
        n = f"origglm_{name}_C{C:g}"
        auc = roc_auc_score(tr[TARGET].astype(int), o)
        print(n, "orig-cv", round(cv, 5), "train AUC", round(auc, 5), flush=True)
        save_predictions(RESULTS_DIR, n, oof=o, test=t, train_ids=tr["id"].to_numpy(), test_ids=te["id"].to_numpy())
        save_cv_result(RESULTS_DIR, n, [], auc, metric_name="auc", params={"C": C, "num": name})
