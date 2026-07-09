"""Shared CV plumbing for PS S6E7 train_*.py scripts.

Keeps the row-order invariant, label<->index mapping, decision-weight tuning and
artifact saving in ONE place so every model reports comparable balanced accuracy.
Model-specific bits (params, categorical prep) stay in each train_*.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from cv_results import save_cv_result
from data_science_stuff.kaggle_utils import (
    classification_report_dict,
    print_classification_report,
    tune_decision_weights,
    weighted_predict,
)
from features import CAT_COLS, CLASSES, TARGET, build_features, feature_columns

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_FOLDS = 5
N_CLASSES = len(CLASSES)
# Multi-seed bagging: each seed is a full 5-fold split; OOF is averaged over seeds
# (every row is out-of-fold once per seed), test over all seed*fold fits. Fold-to-fold
# balanced-acc std (~0.0013) exceeds recent LB gains, so this variance reduction is the
# cheapest real lever left at the current plateau.
# S6E7_SEEDS overrides (comma-separated ints) so historical recipes can be reproduced
# exactly, e.g. the v1b champion is S6E7_SEEDS=42. S6E7_RUN_TAG suffixes every artifact
# name in finalize() (oof_{name}{tag}.npy, ...) so reruns never overwrite scored arrays.
SEEDS: list[int] = (
    [int(s) for s in os.environ["S6E7_SEEDS"].split(",")]
    if os.environ.get("S6E7_SEEDS")
    else [42, 7, 123]
)
RUN_TAG: str = os.environ.get("S6E7_RUN_TAG", "")
# S6E7_REPAIR=1 applies the R2a missingness-mechanism repair (see below); gated
# PASS on the re-masked instrument 2026-07-07 (results/diag_repair_r2a.txt).
REPAIR: bool = os.environ.get("S6E7_REPAIR", "") not in ("", "0")

# The 4 columns whose train masks are deterministic trigger rules but whose test
# masks are uniform (results/eda_missingness.txt). Train-only NaN<->trigger
# couplings misfire on ~2/3 of test NaN rows.
MECHANISM_SHIFTED = ["water_intake", "physical_activity_level", "bmi", "diet_type"]


def _uniform_remask(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """R2a repair: iid Bernoulli re-mask of the mechanism-shifted columns.

    Masks observed train values at each column's TEST marginal NaN rate, so the
    training mechanism matches test (uniform NaNs on top of the trigger NaNs)
    and the NaN<->trigger coupling is broken. Applied to the WHOLE train matrix
    (val rows included): every base then produces OOF on the same test-mechanism
    -emulated surface, which is the surface decision weights should be fit on.
    Consequence: repaired OOF scores read ~0.001-0.002 below plain-surface runs
    by construction -- compare repaired runs to repaired runs only.

    The RNG is fixed (not per-seed) so all bases/seeds share one mask and their
    OOF arrays stay blendable on a single surface. Test is NEVER re-masked: it
    already has the uniform mechanism.

    S6E7_REPAIR_MULT scales the re-mask rates (default 1.0 = the gated R2a).
    Train/test marginal NaN rates are ~equal per column, so mult 1.0 roughly
    DOUBLES train NaN volume in the shifted columns vs test (the Day-6 adv-AUC
    0.6886 overshoot); mult<1 trades coupling dilution for volume match.
    """
    mult = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))
    rng = np.random.default_rng(20260707)
    out = train.copy()
    for col in MECHANISM_SHIFTED:
        if col not in out.columns:  # e.g. water dropped via S6E7_DROP_WATER
            continue
        rate = float(test[col].isna().mean()) * mult
        hit = out[col].notna().to_numpy() & (rng.random(len(out)) < rate)
        out.loc[out.index[hit], col] = np.nan
    return out

# fit_fold(train_idx, val_idx, seed, fold) -> (val_proba (n_val, K), test_proba (n_test, K)).
FitFold = Callable[
    [NDArray[np.int64], NDArray[np.int64], int, int],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


@dataclass
class Dataset:
    """Feature matrices + encoded labels, all in build_features() sorted order."""

    train: pd.DataFrame  # feature columns only (+ nothing else)
    test: pd.DataFrame  # feature columns only
    y: NDArray[np.int64]  # encoded target in CLASSES order (0..K-1)
    test_ids: NDArray[np.int64]
    feature_cols: list[str]
    cat_cols: list[str]


def load_dataset() -> Dataset:
    """Load train/test through build_features() and encode the target via CLASSES.

    Encoding through the fixed CLASSES list (not a fitted LabelEncoder) guarantees
    index i == CLASSES[i] for every model, so oof/test probability columns line up
    across LGBM/XGB/CatBoost and decode correctly.
    """
    train_pl = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test_pl = build_features(pl.read_csv(DATA_DIR / "test.csv"))

    observed = set(train_pl[TARGET].unique().to_list())
    assert observed == set(CLASSES), f"target labels {observed} != CLASSES {CLASSES}"

    feature_cols = feature_columns(train_pl)
    cat_cols = [c for c in CAT_COLS if c in feature_cols]

    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    y = train_pd[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    train_X = train_pd[feature_cols].copy()
    test_X = test_pd[feature_cols].copy()
    if REPAIR:
        train_X = _uniform_remask(train_X, test_X)
        print("S6E7_REPAIR=1: R2a uniform re-mask applied to train")
    return Dataset(
        train=train_X,
        test=test_X,
        y=y.astype(np.int64),
        test_ids=test_pd["id"].to_numpy(),
        feature_cols=feature_cols,
        cat_cols=cat_cols,
    )


def bagged_cv(
    ds: Dataset,
    fit_fold: FitFold,
    seeds: list[int] = SEEDS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    """Run 5-fold CV under each seed and average OOF (over seeds) + test (over all fits).

    ``fit_fold`` is a model-specific closure that fits on ``train_idx``, predicts the val
    split and the full test set, and returns their probability arrays. Per-fold argmax
    balanced-acc is collected across all seed*fold fits for the cv_scores log.
    """
    n_train, n_test = len(ds.y), len(ds.test_ids)
    oof = np.zeros((n_train, N_CLASSES))
    test = np.zeros((n_test, N_CLASSES))
    fold_scores: list[float] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros_like(oof)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
            val_proba, test_proba = fit_fold(tr_idx, val_idx, seed, fold)
            oof_seed[val_idx] = val_proba
            test += test_proba / (N_FOLDS * len(seeds))
            score = float(
                balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
            )
            fold_scores.append(score)
            print(f"  seed {seed} fold {fold} balanced_acc (argmax): {score:.4f}")
        oof += oof_seed / len(seeds)
    return oof, test, fold_scores


def _bag_log_weights(
    y_bag: NDArray[np.int64], proba_bag: NDArray[np.float64]
) -> NDArray[np.float64]:
    """One bootstrap bag's log decision weights (module-level for joblib pickling)."""
    return np.asarray(np.log(tune_decision_weights(y_bag, proba_bag)))


def robust_decision_weights(
    y: NDArray[np.int64],
    proba: NDArray[np.float64],
    n_bags: int = 8,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Bootstrap-bagged per-class decision weights (log-space average), normalised to max 1.

    A single Nelder-Mead fit on the full OOF has ~0.004 balanced-acc variance depending on
    the slice it sees (measured); bagging over bootstrap resamples of the OOF stabilises the
    weight vector without touching the shared kaggle_utils helper.

    Bags run in parallel processes (Day-7): the sequential version left 11 cores idle for
    ~30 min per call (8 bags x 15 NM restarts x 690k-row objective evals). Bootstrap
    indices are drawn sequentially first, so results are bit-identical to the old loop.
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(n_bags)]
    log_ws = Parallel(n_jobs=min(n_bags, os.cpu_count() or 1))(
        delayed(_bag_log_weights)(y[idx], proba[idx]) for idx in idxs
    )
    w = np.exp(np.mean(log_ws, axis=0))
    return np.asarray(w / w.max(), dtype=np.float64)


def finalize(
    name: str,
    ds: Dataset,
    oof_proba: NDArray[np.float64],
    test_proba: NDArray[np.float64],
    fold_scores: list[float],
) -> None:
    """Assert alignment, tune decision weights on OOF, log, save arrays + submission.

    ``name`` is the model key: oof_{name}.npy / test_{name}.npy and the cv_scores row.
    S6E7_RUN_TAG (if set) suffixes the key so experimental reruns never overwrite the
    canonical (or previously scored) artifacts.
    """
    name = f"{name}{RUN_TAG}"
    n_train, n_test = len(ds.y), len(ds.test_ids)
    assert oof_proba.shape == (n_train, N_CLASSES), (
        f"oof_proba {oof_proba.shape} != {(n_train, N_CLASSES)}"
    )
    assert test_proba.shape == (n_test, N_CLASSES), (
        f"test_proba {test_proba.shape} != {(n_test, N_CLASSES)}"
    )
    # Probabilities must sum to ~1 across the CLASSES axis (guards a transposed/misordered save).
    assert np.allclose(oof_proba.sum(axis=1), 1.0, atol=1e-3), (
        "oof_proba rows must sum to 1"
    )

    argmax_score = float(balanced_accuracy_score(ds.y, oof_proba.argmax(axis=1)))
    # Bagged (robust) weights are what we deploy; also report the single-shot fit so any
    # divergence between the two is visible (should be wash-or-better).
    single = tune_decision_weights(ds.y, oof_proba)
    weights = robust_decision_weights(ds.y, oof_proba)
    single_score = float(
        balanced_accuracy_score(ds.y, weighted_predict(oof_proba, single))
    )
    weighted_score = float(
        balanced_accuracy_score(ds.y, weighted_predict(oof_proba, weights))
    )

    print(
        f"\n[{name}] OOF balanced_acc  argmax={argmax_score:.4f}  "
        f"weighted(single)={single_score:.4f}  weighted(bagged)={weighted_score:.4f}"
    )
    print_classification_report(
        classification_report_dict(ds.y, oof_proba, weights=weights, labels=CLASSES)
    )
    print(f"decision weights: {dict(zip(CLASSES, weights.round(4)))}")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_cv_result(
        RESULTS_DIR, name, fold_scores, weighted_score, metric_name="balanced_acc"
    )
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof_proba)
    np.save(RESULTS_DIR / f"test_{name}.npy", test_proba)
    with (RESULTS_DIR / f"decision_weights_{name}.json").open("w") as f:
        json.dump(dict(zip(CLASSES, weights.tolist())), f, indent=2)
    print(f"arrays + weights saved → {RESULTS_DIR}")

    # Standalone submission (decision-weighted) as a per-model sanity check.
    test_pred = weighted_predict(test_proba, weights)
    labels = np.array(CLASSES)[test_pred]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"{name}.csv"
    pd.DataFrame({"id": ds.test_ids, TARGET: labels}).to_csv(out_path, index=False)
    print(f"submission saved → {out_path}")
