"""Blend LGBM + LGBM-AR + CatBoost + XGBoost + MLP OOF predictions and build final submission."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TARGET = "PitNextLap"

MODELS = ["lgbm", "lgbm_ar", "catboost", "xgboost", "mlp"]


def neg_ensemble_auc(
    weights: np.ndarray,
    oofs: list[np.ndarray],
    y: np.ndarray,
) -> float:
    w = np.clip(weights, 0, None)
    w = w / w.sum()
    blend = sum(wi * oof for wi, oof in zip(w, oofs))
    return -float(roc_auc_score(y, blend))


def optimize_weights(
    oofs: list[np.ndarray], y: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    x0 = np.ones(len(oofs)) / len(oofs)
    result = minimize(
        neg_ensemble_auc,
        x0,
        args=([o[mask] for o in oofs], y[mask]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train = build_features(train_raw)
    train = compute_group_features(train_raw, train)
    y = train[TARGET].to_numpy()
    # Must go through build_features so IDs are in the same sorted order
    # as the test_{model}.npy prediction arrays.
    test_df = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    test_ids = test_df["id"].to_numpy()
    is_2023_train = (train["Year"] == 2023).to_numpy().astype(bool)
    is_2023_test = (test_df["Year"] == 2023).to_numpy().astype(bool)

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

    # --- Strategy 1: simple average ---
    simple_blend_oof = np.mean(oofs, axis=0)
    simple_auc = float(roc_auc_score(y, simple_blend_oof))
    print(f"\n  Simple average OOF AUC: {simple_auc:.4f}")

    # --- Strategy 2: flat optimized weights ---
    opt_w = optimize_weights(oofs, y, mask=np.ones(len(y), dtype=bool))
    opt_blend_oof = sum(w * oof for w, oof in zip(opt_w, oofs))
    opt_auc = float(roc_auc_score(y, opt_blend_oof))
    print(f"  Flat optimized OOF AUC: {opt_auc:.4f}")
    for name, w in zip(MODELS[: len(oofs)], opt_w):
        print(f"    {name}: {w:.3f}")

    # --- Strategy 3: conditional weights split on is_2023 ---
    print(f"\n  Optimizing 2023 weights  (n={is_2023_train.sum()}) ...")
    w_2023 = optimize_weights(oofs, y, mask=is_2023_train)
    print(f"  Optimizing non-2023 weights (n={(~is_2023_train).sum()}) ...")
    w_non2023 = optimize_weights(oofs, y, mask=~is_2023_train)

    oof_cond = np.empty(len(y))
    oof_cond[is_2023_train] = sum(
        w * o[is_2023_train] for w, o in zip(w_2023, oofs)
    )
    oof_cond[~is_2023_train] = sum(
        w * o[~is_2023_train] for w, o in zip(w_non2023, oofs)
    )
    cond_auc = float(roc_auc_score(y, oof_cond))
    print(f"  Conditional OOF AUC: {cond_auc:.4f}")
    print("  2023 weights:")
    for name, w in zip(MODELS[: len(oofs)], w_2023):
        print(f"    {name}: {w:.3f}")
    print("  non-2023 weights:")
    for name, w in zip(MODELS[: len(oofs)], w_non2023):
        print(f"    {name}: {w:.3f}")

    # --- Pick best strategy ---
    candidates = [
        ("simple_avg",   simple_auc, simple_blend_oof, np.mean(tests, axis=0)),
        ("flat_opt",     opt_auc,    opt_blend_oof,    sum(w * t for w, t in zip(opt_w, tests))),
        ("conditional",  cond_auc,   oof_cond,         None),
    ]
    test_cond = np.empty(len(test_ids))
    test_cond[is_2023_test] = sum(
        w * t[is_2023_test] for w, t in zip(w_2023, tests)
    )
    test_cond[~is_2023_test] = sum(
        w * t[~is_2023_test] for w, t in zip(w_non2023, tests)
    )
    # patch the None placeholder
    candidates[2] = ("conditional", cond_auc, oof_cond, test_cond)

    strategy, best_auc, best_oof, best_test = max(candidates, key=lambda t: t[1])
    print(f"\nBest strategy: {strategy}  OOF AUC: {best_auc:.4f}")

    save_cv_result(RESULTS_DIR, "ensemble_v4", [], best_auc)
    np.save(RESULTS_DIR / "oof_ensemble.npy", best_oof)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: best_test})
    out_path = SUBMISSIONS_DIR / "ensemble_v4.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
