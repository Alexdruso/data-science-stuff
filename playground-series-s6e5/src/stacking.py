"""Logistic regression stacking meta-learner for PS S6E5 — F1 Pit Stop Prediction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
MODELS = ["lgbm", "catboost", "xgboost", "mlp"]
EPS = 1e-7
Cs = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def main() -> None:
    oofs = [np.load(RESULTS_DIR / f"oof_{m}.npy") for m in MODELS]
    tests = [np.load(RESULTS_DIR / f"test_{m}.npy") for m in MODELS]

    # Must go through build_features so row order matches the sorted OOF arrays
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    y = build_features(train_raw)[TARGET].to_numpy()
    test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

    for m, oof in zip(MODELS, oofs):
        print(f"  {m:10s} OOF AUC: {roc_auc_score(y, oof):.4f}")

    X_meta = np.column_stack([logit(oof) for oof in oofs])
    X_test_meta = np.column_stack([logit(test) for test in tests])
    print(f"\nMeta-feature matrix: {X_meta.shape}")

    # Nested CV: 10-fold outer, 5-fold inner for C selection
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_stack = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y), 1):
        clf = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            scoring="roc_auc",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_meta[train_idx], y[train_idx])
        oof_stack[val_idx] = clf.predict_proba(X_meta[val_idx])[:, 1]
        fold_auc = float(roc_auc_score(y[val_idx], oof_stack[val_idx]))
        print(f"  Fold {fold:2d} AUC: {fold_auc:.4f}  (C={clf.C_[0]:.4f})")

    oof_auc = float(roc_auc_score(y, oof_stack))
    print(f"\nStacking OOF AUC: {oof_auc:.4f}")

    # Final model on all data
    clf_final = LogisticRegressionCV(
        Cs=Cs,
        cv=10,
        scoring="roc_auc",
        max_iter=1000,
        random_state=42,
    )
    clf_final.fit(X_meta, y)
    print(f"Final C: {clf_final.C_[0]:.4f}")
    print("Coefficients:")
    for name, coef in zip(MODELS, clf_final.coef_[0]):
        print(f"  {name:10s}: {coef:.4f}")

    test_stack = clf_final.predict_proba(X_test_meta)[:, 1]

    save_cv_result(RESULTS_DIR, "stacking_v1", [], oof_auc)
    np.save(RESULTS_DIR / "oof_stacking.npy", oof_stack)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_stack})
    out_path = SUBMISSIONS_DIR / "stacking_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}  ({len(submission)} rows)")


if __name__ == "__main__":
    main()
