"""Nelder-Mead ensemble blending for PS S6E6.

Optimises blend weights over OOF probability arrays to maximise balanced accuracy.
Applies the same weights to test probability arrays to generate a submission.

Usage:
    python src/ensemble.py

Expected files in results/:
    oof_lgbm.npy        (n_train × 3)
    oof_xgboost.npy     (n_train × 3)
    oof_catboost.npy    (n_train × 3)
    test_lgbm.npy       (n_test × 3)
    test_xgboost.npy    (n_test × 3)
    test_catboost.npy   (n_test × 3)

Any subset that exists will be included; missing models are skipped with a warning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.blending import (
    blend,
    diversity_report,
    optimize_blend_weights,
)
from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

MODELS = ["lgbm", "xgboost", "catboost", "mlp", "realmlp", "tabnet", "extratrees", "rf", "logreg", "knn", "nb"]
# Anchor model the diversity report compares every other model against.
ANCHOR = "lgbm"


def load_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for m in MODELS:
        oof_path = RESULTS_DIR / f"oof_{m}.npy"
        test_path = RESULTS_DIR / f"test_{m}.npy"
        if oof_path.exists() and test_path.exists():
            oof[m] = np.load(oof_path)
            test[m] = np.load(test_path)
            print(f"  Loaded {m}: oof {oof[m].shape}  test {test[m].shape}")
        else:
            print(f"  Skipping {m} — files not found")
    return oof, test


def balanced_acc_argmax(y: np.ndarray, blended: np.ndarray) -> float:
    """Blend metric: balanced accuracy on argmax (higher is better)."""
    return float(balanced_accuracy_score(y, np.argmax(blended, axis=1)))


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train_pl = build_features(train_raw)
    test_pl = build_features(pl.read_csv(DATA_DIR / "test.csv"))

    le = LabelEncoder()
    y = le.fit_transform(train_pl[TARGET].to_numpy())
    test_ids = test_pl["id"].to_numpy()

    print("Loading OOF/test arrays:")
    oof_arrays, test_arrays = load_arrays()

    if len(oof_arrays) < 2:
        raise SystemExit("Need at least 2 models to blend — run more training scripts first.")

    model_names = list(oof_arrays.keys())
    oof_list = [oof_arrays[m] for m in model_names]
    test_list = [test_arrays[m] for m in model_names]

    # Individual model scores
    print("\nIndividual OOF balanced_acc:")
    for m in model_names:
        score = balanced_accuracy_score(y, np.argmax(oof_arrays[m], axis=1))
        print(f"  {m}: {score:.4f}")

    if ANCHOR in oof_arrays:
        print(f"\nDiversity report (anchor = {ANCHOR}):")
        report = diversity_report(
            oof_arrays, y, le.classes_.tolist(), anchor=ANCHOR
        )
        print(report.round(4).to_string())

    # Optimise blend weights (softmax parameterization, so they're unconstrained)
    w = optimize_blend_weights(oof_list, y, balanced_acc_argmax, normalize="softmax")
    print(f"\nOptimal weights: {dict(zip(model_names, w.round(4)))}")

    # Evaluate raw blend
    blended_oof = blend(oof_list, w)
    oof_pred = np.argmax(blended_oof, axis=1)
    oof_score = float(balanced_accuracy_score(y, oof_pred))
    print(f"Ensemble OOF balanced_acc (argmax): {oof_score:.4f}")

    # Threshold weight optimisation on blended OOF
    tw, best_score = optimize_thresholds(blended_oof, y)
    print(f"Ensemble OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    print(f"Threshold weights: {dict(zip(le.classes_, tw.round(4)))}")
    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / "threshold_weights_ensemble.json")

    save_cv_result(RESULTS_DIR, "ensemble", [], best_score, metric_name="balanced_acc")
    np.save(RESULTS_DIR / "oof_ensemble.npy", blended_oof)

    # Test predictions
    blended_test = blend(test_list, w)
    np.save(RESULTS_DIR / "test_ensemble.npy", blended_test)

    test_pred_labels = le.inverse_transform(np.argmax(blended_test * tw, axis=1))
    run_name = f"ensemble_{'_'.join(model_names)}"
    out_path = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", test_ids, TARGET, test_pred_labels)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
