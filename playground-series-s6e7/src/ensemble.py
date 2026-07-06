"""Two-stage Nelder-Mead ensemble for PS S6E7 — Predicting Student Health Risk.

Stage 1: optimise blend weights over OOF probability arrays (balanced accuracy).
Stage 2: optimise per-class decision weights on the blended OOF (prior-corrected
argmax), reusing data_science_stuff.kaggle_utils.tune_decision_weights.

Both stages score through balanced_accuracy_score so numbers are comparable with
the per-model reports. Any subset of models present in results/ is blended.

Usage: python src/ensemble.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from data_science_stuff.kaggle_utils import weighted_predict
from features import CLASSES, TARGET, build_features
from train_common import robust_decision_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = ["lgbm", "xgboost", "catboost", "mlp", "realmlp", "tabm"]
ANCHOR = "lgbm"  # diversity report compares every model against this one


def load_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for m in MODELS:
        oof_path, test_path = (
            RESULTS_DIR / f"oof_{m}.npy",
            RESULTS_DIR / f"test_{m}.npy",
        )
        if oof_path.exists() and test_path.exists():
            oof[m], test[m] = np.load(oof_path), np.load(test_path)
            print(f"  Loaded {m}: oof {oof[m].shape}  test {test[m].shape}")
        else:
            print(f"  Skipping {m} — files not found")
    return oof, test


def precorrect(
    oof: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Scale each model's probs by its OWN saved per-class decision weights, renormalised.

    The blend objective below argmaxes the blended probabilities, but each model's
    balanced-accuracy-optimal decision surface is argmax(proba * w_model), not
    argmax(proba). Blending raw probs therefore discards models (LGBM/XGB here) whose
    plain argmax is poor but whose corrected argmax is competitive — collapsing the
    blend onto whichever model happens to be self-calibrated (CatBoost). Pre-correcting
    each model with its own weights puts every model on its deployed decision surface
    before blending; a final decision tune still runs on the blend.
    """
    oc: dict[str, np.ndarray] = {}
    tc: dict[str, np.ndarray] = {}
    for m in oof:
        wpath = RESULTS_DIR / f"decision_weights_{m}.json"
        if wpath.exists():
            with wpath.open() as f:
                wd = json.load(f)
            w = np.array([wd[c] for c in CLASSES])
            oc[m] = (oof[m] * w) / (oof[m] * w).sum(axis=1, keepdims=True)
            tc[m] = (test[m] * w) / (test[m] * w).sum(axis=1, keepdims=True)
        else:
            oc[m], tc[m] = oof[m], test[m]
    return oc, tc


def blend(weights: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    """Weighted average of probability arrays; weights are softmax-normalised."""
    w = np.exp(weights) / np.exp(weights).sum()
    return sum(w[i] * a for i, a in enumerate(arrays))  # type: ignore[return-value]


def objective(weights: np.ndarray, oof_list: list[np.ndarray], y: np.ndarray) -> float:
    return -float(
        balanced_accuracy_score(y, np.argmax(blend(weights, oof_list), axis=1))
    )


def diversity_report(oof_arrays: dict[str, np.ndarray], y: np.ndarray) -> None:
    """Per-class recall + error-overlap vs the anchor: does a model fix DIFFERENT errors?"""
    if ANCHOR not in oof_arrays:
        return
    anchor_pred = np.argmax(oof_arrays[ANCHOR], axis=1)
    anchor_wrong = anchor_pred != y
    print(f"\nDiversity report (anchor = {ANCHOR}):")
    print(
        f"  {'model':<10} "
        + " ".join(f"rec_{c[:4]}" for c in CLASSES)
        + "   fixes_anchor  anchor_fixes"
    )
    for m, arr in oof_arrays.items():
        pred = np.argmax(arr, axis=1)
        rec = recall_score(y, pred, average=None, labels=list(range(len(CLASSES))))
        rec_str = " ".join(f"{r:.4f}" for r in rec)
        if m == ANCHOR:
            print(f"  {m:<10} {rec_str}   (anchor)")
            continue
        m_wrong = pred != y
        fixes = float((pred[anchor_wrong] == y[anchor_wrong]).mean())
        anchor_fixes = float((anchor_pred[m_wrong] == y[m_wrong]).mean())
        print(f"  {m:<10} {rec_str}   {fixes:>8.3f}     {anchor_fixes:>8.3f}")


def main() -> None:
    train_pl = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test_pl = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    y = (
        train_pl[TARGET]
        .map_elements({c: i for i, c in enumerate(CLASSES)}.get, return_dtype=pl.Int64)
        .to_numpy()
    )
    test_ids = test_pl["id"].to_numpy()

    print("Loading OOF/test arrays:")
    oof_arrays, test_arrays = load_arrays()
    if len(oof_arrays) < 2:
        raise SystemExit(
            "Need at least 2 models to blend — run more training scripts first."
        )

    # Put every model on its deployed (decision-weighted) surface before blending.
    oof_arrays, test_arrays = precorrect(oof_arrays, test_arrays)

    names = list(oof_arrays.keys())
    oof_list = [oof_arrays[m] for m in names]
    test_list = [test_arrays[m] for m in names]

    print("\nIndividual OOF balanced_acc (argmax):")
    for m in names:
        print(
            f"  {m}: {balanced_accuracy_score(y, np.argmax(oof_arrays[m], axis=1)):.4f}"
        )
    diversity_report(oof_arrays, y)

    # ── stage 1: blend weights ───────────────────────────────────────────────
    result = minimize(
        objective,
        np.ones(len(names)),
        args=(oof_list, y),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    w = np.exp(result.x) / np.exp(result.x).sum()
    print(f"\nBlend weights: {dict(zip(names, w.round(4)))}")

    blended_oof = blend(result.x, oof_list)
    blended_test = blend(result.x, test_list)
    argmax_score = float(balanced_accuracy_score(y, np.argmax(blended_oof, axis=1)))
    print(f"Ensemble OOF balanced_acc (argmax): {argmax_score:.4f}")

    # ── stage 2: per-class decision weights on blended OOF ───────────────────
    dw = robust_decision_weights(y, blended_oof)
    weighted_score = float(
        balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    )
    print(f"Ensemble OOF balanced_acc (decision-tuned): {weighted_score:.4f}")
    print(f"Decision weights: {dict(zip(CLASSES, dw.round(4)))}")

    with (RESULTS_DIR / "decision_weights_ensemble.json").open("w") as f:
        json.dump(
            {
                "blend": dict(zip(names, w.tolist())),
                "decision": dict(zip(CLASSES, dw.tolist())),
            },
            f,
            indent=2,
        )
    save_cv_result(
        RESULTS_DIR, "ensemble", [], weighted_score, metric_name="balanced_acc"
    )
    np.save(RESULTS_DIR / "oof_ensemble.npy", blended_oof)
    np.save(RESULTS_DIR / "test_ensemble.npy", blended_test)

    # ── submission ───────────────────────────────────────────────────────────
    labels = np.array(CLASSES)[weighted_predict(blended_test, dw)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "ensemble_v1.csv"
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}  ({len(labels)} rows)")
    print(f"  label distribution: {dict(pd.Series(labels).value_counts())}")


if __name__ == "__main__":
    main()
