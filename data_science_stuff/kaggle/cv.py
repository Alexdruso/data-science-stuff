"""The standard cross-validation loop shared by every ``train_<model>.py``.

Every competition trainer repeats the same skeleton: ``StratifiedKFold`` with
``random_state=42`` (fold alignment across models is what makes OOF arrays
blendable), ``oof[val_idx] = val_pred``, ``test += test_pred / n_splits``.
:func:`run_cv` owns that skeleton; the caller keeps the model, metric,
saving, and submission logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from data_science_stuff.kaggle.blending import ScoreFn

Matrix = Union[pd.DataFrame, "NDArray[Any]"]

# fit_fold(X_tr, y_tr, X_va, y_va, X_test, fold) -> (val_pred, test_pred).
# Per-fold transforms that must not leak (fold-aware target encoding,
# scalers fit on the training split) belong INSIDE fit_fold.
FitFoldFn = Callable[
    ["Matrix", "NDArray[Any]", "Matrix", "NDArray[Any]", "Matrix", int],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


class CVResult(NamedTuple):
    """Out-of-fold predictions, mean test predictions, per-fold scores."""

    oof: NDArray[np.float64]
    test: NDArray[np.float64]
    fold_scores: list[float]


def _take(x: Matrix, idx: NDArray[Any]) -> Matrix:
    if isinstance(x, pd.DataFrame):
        taken: pd.DataFrame = x.iloc[idx]
        return taken
    subset: NDArray[Any] = x[idx]
    return subset


def run_cv(
    X: Matrix,
    y: NDArray[Any],
    X_test: Matrix,
    fit_fold: FitFoldFn,
    *,
    n_splits: int = 5,
    seed: int = 42,
    n_outputs: int = 1,
    stratified: bool = True,
    score_fn: ScoreFn | None = None,
    after_fold: Callable[[int], None] | None = None,
) -> CVResult:
    """Run the standard K-fold loop and return OOF/test predictions.

    Args:
        X: Training features (DataFrame or array), in ``build_features()``
            sort order — the row-order invariant applies to the outputs.
        y: Training target, same order.
        X_test: Test features, passed to ``fit_fold`` every fold.
        fit_fold: ``(X_tr, y_tr, X_va, y_va, X_test, fold) ->
            (val_pred, test_pred)``. Fit the model (and any fold-aware
            transforms) on the training split only.
        n_splits: Number of folds.
        seed: ``random_state`` for the splitter. Keep the default 42 so OOF
            folds stay aligned across models (required for stacking).
        n_outputs: 1 for 1-D predictions (binary proba / regression);
            n_classes for multiclass probability arrays.
        stratified: ``StratifiedKFold`` when True (classification),
            ``KFold`` otherwise (regression).
        score_fn: Optional ``(y_val, val_pred) -> float``; when given,
            per-fold scores are collected in ``CVResult.fold_scores``.
        after_fold: Optional hook called with the fold number after each
            fold — the place for ``torch.cuda.empty_cache()`` (GPU memory
            rule: ``fit_fold`` must also ``del`` its GPU tensors/models
            before returning).

    Returns:
        :class:`CVResult` with ``oof`` (n_train,) or (n_train, n_outputs),
        ``test`` averaged over folds, and ``fold_scores``.
    """
    n_train, n_test = len(X), len(X_test)
    oof = np.zeros(n_train if n_outputs == 1 else (n_train, n_outputs))
    test = np.zeros(n_test if n_outputs == 1 else (n_test, n_outputs))
    fold_scores: list[float] = []

    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(n_train), y), 1):
        val_pred, test_pred = fit_fold(
            _take(X, tr_idx), y[tr_idx], _take(X, va_idx), y[va_idx], X_test, fold
        )
        oof[va_idx] = val_pred
        test += np.asarray(test_pred, dtype=np.float64) / n_splits
        if score_fn is not None:
            fold_scores.append(float(score_fn(y[va_idx], np.asarray(val_pred))))
        if after_fold is not None:
            after_fold(fold)
    return CVResult(oof, test, fold_scores)
