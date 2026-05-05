"""Blend LGBM + CatBoost + XGBoost OOF predictions and build final submission."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TARGET = "PitNextLap"

MODELS = ["lgbm", "catboost", "xgboost"]


def neg_ensemble_auc(
    weights: np.ndarray,
    oofs: list[np.ndarray],
    y: np.ndarray,
) -> float:
    w = np.clip(weights, 0, None)
    w = w / w.sum()
    blend = sum(wi * oof for wi, oof in zip(w, oofs))
    return -float(roc_auc_score(y, blend))


def main() -> None:
    train = pl.read_csv(DATA_DIR / "train.csv")
    y = train[TARGET].to_numpy()
    test_ids = pl.read_csv(DATA_DIR / "test.csv")["id"].to_numpy()

    oofs, tests = [], []
    for model in MODELS:
        oof_path = RESULTS_DIR / f"oof_{model}.npy"
        test_path = RESULTS_DIR / f"test_{model}.npy"
        if not oof_path.exists() or not test_path.exists():
            print(f"Missing predictions for {model} — skipping")
            continue
        oofs.append(np.load(oof_path))
        tests.append(np.load(test_path))
        auc = roc_auc_score(y, oofs[-1])
        print(f"  {model:10s} OOF AUC: {auc:.4f}")

    if len(oofs) < 2:
        print("Need at least 2 models to ensemble.")
        return

    # Simple average
    simple_blend_oof = np.mean(oofs, axis=0)
    simple_auc = float(roc_auc_score(y, simple_blend_oof))
    print(f"\n  Simple average OOF AUC: {simple_auc:.4f}")

    # Optimized weights (Nelder-Mead, non-negative)
    x0 = np.ones(len(oofs)) / len(oofs)
    result = minimize(
        neg_ensemble_auc,
        x0=x0,
        args=(oofs, y),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    opt_w = np.clip(result.x, 0, None)
    opt_w = opt_w / opt_w.sum()
    opt_blend_oof = sum(w * oof for w, oof in zip(opt_w, oofs))
    opt_auc = float(roc_auc_score(y, opt_blend_oof))
    print(f"  Optimized weights OOF AUC: {opt_auc:.4f}")
    for name, w in zip(MODELS[: len(oofs)], opt_w):
        print(f"    {name}: {w:.3f}")

    # Pick best blend
    if opt_auc >= simple_auc:
        best_oof = opt_blend_oof
        best_test = sum(w * t for w, t in zip(opt_w, tests))
        best_auc = opt_auc
        strategy = "optimized"
    else:
        best_oof = simple_blend_oof
        best_test = np.mean(tests, axis=0)
        best_auc = simple_auc
        strategy = "simple_avg"

    print(f"\nBest strategy: {strategy}  OOF AUC: {best_auc:.4f}")

    model_names = "_".join(MODELS[: len(oofs)])
    run_name = f"ensemble_{model_names}_v1"
    save_cv_result(RESULTS_DIR, run_name, [], best_auc)

    np.save(RESULTS_DIR / "oof_ensemble.npy", best_oof)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: best_test})
    out_path = SUBMISSIONS_DIR / "ensemble_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
