"""Stacking v2 — LR meta-learner with OOF predictions + original feature matrix.

Fixed C=1.0 per fold (no inner CV). For a meta-learner on calibrated base-model
predictions the exact C matters very little; eliminating the inner CV makes each
fold run in seconds rather than hours.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
MODELS = ["lgbm", "catboost", "xgboost", "mlp"]
DROP_COLS: frozenset[str] = frozenset({"id", TARGET, "Driver"})
CAT_COLS: list[str] = ["Compound", "Race"]
EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe: Pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe: Pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )


def run_variant(
    label: str,
    C: float,
    penalty: str,
    X_logit: np.ndarray,
    X_orig_df: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))

    print(f"\n=== {label} (C={C}, penalty={penalty}) ===", flush=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_logit, y), 1):
        prep = make_preprocessor(num_cols, cat_cols)
        X_train_orig = prep.fit_transform(X_orig_df.iloc[train_idx])
        X_val_orig = prep.transform(X_orig_df.iloc[val_idx])

        X_train = np.hstack([X_logit[train_idx], X_train_orig])
        X_val = np.hstack([X_logit[val_idx], X_val_orig])

        solver = "lbfgs" if penalty == "l2" else "liblinear"
        clf = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=200,
            tol=1e-3,
            random_state=42,
        )
        clf.fit(X_train, y[train_idx])
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        fold_auc = float(roc_auc_score(y[val_idx], oof[val_idx]))
        print(f"  Fold {fold:2d} AUC: {fold_auc:.4f}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"  {label} OOF AUC: {auc:.4f}", flush=True)
    return oof, auc


def main() -> None:
    oofs = [np.load(RESULTS_DIR / f"oof_{m}.npy") for m in MODELS]
    tests = [np.load(RESULTS_DIR / f"test_{m}.npy") for m in MODELS]

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")

    train_df = build_features(train_raw)
    test_df = build_features(test_raw)
    train_df = compute_group_features(train_raw, train_df)
    test_df = compute_group_features(train_raw, test_df)

    y = train_df[TARGET].to_numpy()
    test_ids = test_df["id"].to_numpy()

    print("Base model OOF AUCs:", flush=True)
    for m, oof in zip(MODELS, oofs):
        print(f"  {m:10s}: {roc_auc_score(y, oof):.4f}", flush=True)

    X_logit = np.column_stack([logit(oof) for oof in oofs])
    X_test_logit = np.column_stack([logit(t) for t in tests])
    print(f"\nLogit meta-features: {X_logit.shape}", flush=True)

    keep_cols = [c for c in train_df.columns if c not in DROP_COLS]
    train_pd = train_df.select(keep_cols).to_pandas()
    test_keep = [c for c in keep_cols if c in test_df.columns]
    test_pd = test_df.select(test_keep).to_pandas()
    for col in keep_cols:
        if col not in test_pd.columns:
            test_pd[col] = 0.0
    test_pd = test_pd[keep_cols]

    num_cols = [c for c in keep_cols if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in keep_cols]
    print(f"Orig features: {len(num_cols)} numeric + {len(cat_cols)} categorical", flush=True)

    # Try L2 at three C values and L1 at C=1.0
    results: list[tuple[str, float, np.ndarray]] = []
    for C in [0.1, 1.0, 10.0]:
        oof, auc = run_variant(f"L2 C={C}", C, "l2", X_logit, train_pd, y, num_cols, cat_cols)
        results.append((f"L2_C{C}", auc, oof))
    oof_l1, auc_l1 = run_variant("L1 C=1.0", 1.0, "l1", X_logit, train_pd, y, num_cols, cat_cols)
    results.append(("L1_C1.0", auc_l1, oof_l1))

    best_label, best_auc, best_oof = max(results, key=lambda t: t[1])
    print(f"\nBest: {best_label}  OOF AUC: {best_auc:.4f}", flush=True)

    # Final model on all training data using best C/penalty
    best_C = float(best_label.split("C")[1])
    best_penalty = "l2" if best_label.startswith("L2") else "l1"
    prep_final = make_preprocessor(num_cols, cat_cols)
    X_train_full = np.hstack([X_logit, prep_final.fit_transform(train_pd)])
    X_test_full = np.hstack([X_test_logit, prep_final.transform(test_pd)])

    final_solver = "lbfgs" if best_penalty == "l2" else "liblinear"
    clf_final = LogisticRegression(
        C=best_C,
        penalty=best_penalty,
        solver=final_solver,
        max_iter=200,
        tol=1e-3,
        random_state=42,
    )
    clf_final.fit(X_train_full, y)

    test_preds = clf_final.predict_proba(X_test_full)[:, 1]

    save_cv_result(RESULTS_DIR, "stacking_v2", [], best_auc)
    np.save(RESULTS_DIR / "oof_stacking_v2.npy", best_oof)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_preds})
    out_path = SUBMISSIONS_DIR / "stacking_v2.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}  ({len(submission)} rows)", flush=True)


if __name__ == "__main__":
    main()
