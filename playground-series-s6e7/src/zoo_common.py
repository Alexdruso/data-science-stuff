"""Shared harness for the Day-8 architecture-diversity zoo (PS S6E7).

Every zoo candidate is one seed x 5 folds on the REPAIRED surface (run with
S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42) so its OOF is gateable against
the deployed core `oof_ensemble_r_breadth.npy` by diag_mlp_transfer.py. Judged
by decorrelation signature (fix-share / test-like fixes / error-overlap), not
blend OOF — user ruling 2026-07-07.

Provides:
- zoo_cv(): bagged_cv-equivalent single-seed fold loop with per-fold checkpoints
  (this box hard-reboots at random; a lost run must resume, not restart).
- te_block_for_fold(): the deferred-for-NN input recipe — exact-value TE of the
  6 numerics + rule-combo TE (stress x activity x sleep_quality) + 4 ordinal
  scalars — inner-cross-fitted per fold (invariant #3), cached per (tag, seed,
  fold) so all zoo candidates share one computation.
- fold_impute_stats() / apply_impute(): train-fold median impute + z-score for
  models that cannot take NaN (reused semantics of train_mlp._standardize).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS
from train_common import N_CLASSES, N_FOLDS, RESULTS_DIR, Dataset
from train_fe import ORDINAL, combo_key
from train_te_num import encode_fold, with_te

CACHE_DIR = RESULTS_DIR / "cache"

FitFold = Callable[
    [NDArray[np.int64], NDArray[np.int64], int, int],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


def zoo_cv(
    ds: Dataset,
    fit_fold: FitFold,
    ckpt_name: str,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    """Single-seed 5-fold CV, identical splits to train_common.bagged_cv(seeds=[seed]),
    with per-fold val/test checkpoints so a mid-run reboot resumes instead of restarting."""
    ckpt = CACHE_DIR / f"ckpt_{ckpt_name}"
    ckpt.mkdir(parents=True, exist_ok=True)
    n_train, n_test = len(ds.y), len(ds.test_ids)
    oof = np.zeros((n_train, N_CLASSES))
    test = np.zeros((n_test, N_CLASSES))
    fold_scores: list[float] = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        vck, tck = ckpt / f"f{fold}_val.npy", ckpt / f"f{fold}_test.npy"
        if vck.exists() and tck.exists():
            val_proba, test_proba = np.load(vck), np.load(tck)
            print(f"  fold {fold}: loaded checkpoint")
        else:
            val_proba, test_proba = fit_fold(tr_idx, val_idx, seed, fold)
            np.save(vck, val_proba)
            np.save(tck, test_proba)
        oof[val_idx] = val_proba
        test += test_proba / N_FOLDS
        score = float(balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1)))
        fold_scores.append(score)
        print(f"  seed {seed} fold {fold} balanced_acc (argmax): {score:.4f}")
    return oof, test, fold_scores


def clear_ckpt(ckpt_name: str) -> None:
    ckpt = CACHE_DIR / f"ckpt_{ckpt_name}"
    if ckpt.exists():
        for f in ckpt.glob("f*.npy"):
            f.unlink()
        ckpt.rmdir()


COMBO_COL = "_combo"


def te_block_for_fold(
    ds: Dataset,
    tr_idx: NDArray[np.int64],
    seed: int,
    fold: int,
    tag: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str]]:
    """The NN input recipe for one fold, cached: (train_full, test, names).

    Columns: te_<num>_{0,1,2} for the 6 numerics + te__combo_{0,1,2} (rule-combo)
    — training rows inner-cross-fitted, val/test rows from full-fold maps
    (train_te_num protocol) — plus the 4 ordinal scalars (fold-independent).
    TE NaN (source value missing) is filled with the fold class prior — the
    marginalized posterior — so NN consumers get a NaN-free block; the ordinals
    keep NaN (consumers impute; missingness indicators live in NNData).

    `tag` must identify the surface (e.g. "r" for REPAIR=1 mult 1.0): TE maps
    depend on which rows are NaN, so surfaces must never share a cache entry.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"te_{tag}_s{seed}_f{fold}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["train"], z["test"], list(z["names"])

    num_cols = [c for c in NUM_COLS if c in ds.train.columns]
    src_tr = ds.train[num_cols].copy()
    src_te = ds.test[num_cols].copy()
    src_tr[COMBO_COL] = combo_key(ds.train)
    src_te[COMBO_COL] = combo_key(ds.test)
    cols = num_cols + [COMBO_COL]

    te_tr, full_maps, prior = encode_fold(src_tr, ds.y, tr_idx, cols, seed)
    te_names = [f"te_{c}_{k}" for c in cols for k in range(N_CLASSES)]
    train_block = np.full((len(ds.y), len(te_names)), np.nan, dtype=np.float32)
    train_block[tr_idx] = te_tr[te_names].to_numpy(dtype=np.float32)
    val_mask = np.ones(len(ds.y), dtype=bool)
    val_mask[tr_idx] = False
    val_te = with_te(src_tr[val_mask], full_maps, prior)
    train_block[val_mask] = val_te[te_names].to_numpy(dtype=np.float32)
    test_block = with_te(src_te, full_maps, prior)[te_names].to_numpy(dtype=np.float32)

    # fill TE NaN (source missing) with the fold prior = marginalized posterior
    prior_row = np.tile(prior.astype(np.float32), len(cols))
    for block in (train_block, test_block):
        nan_r, nan_c = np.where(np.isnan(block))
        block[nan_r, nan_c] = prior_row[nan_c]

    ord_tr = _ordinal_block(ds.train)
    ord_te = _ordinal_block(ds.test)
    train_out = np.concatenate([train_block, ord_tr], axis=1)
    test_out = np.concatenate([test_block, ord_te], axis=1)
    names = te_names + [f"{c}_ord" for c in ORDINAL]
    np.savez_compressed(cache, train=train_out, test=test_out, names=np.array(names))
    return train_out, test_out, names


def _ordinal_block(df: pd.DataFrame) -> NDArray[np.float32]:
    out = np.full((len(df), len(ORDINAL)), np.nan, dtype=np.float32)
    for i, (col, m) in enumerate(ORDINAL.items()):
        out[:, i] = df[col].map(m).to_numpy(dtype=np.float32)
    return out


def fold_impute_stats(
    X: NDArray[np.float64], tr_idx: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Train-fold median + mean/std for NaN-free consumers (train_mlp semantics)."""
    med = np.nanmedian(X[tr_idx], axis=0)
    filled = np.where(np.isnan(X), med, X)
    mean = filled[tr_idx].mean(axis=0)
    std = filled[tr_idx].std(axis=0) + 1e-6
    return med, mean, std


def apply_impute(
    X: NDArray[np.float64],
    med: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
) -> NDArray[np.float32]:
    return (((np.where(np.isnan(X), med, X)) - mean) / std).astype(np.float32)
