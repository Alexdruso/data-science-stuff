"""The standard cross-validation loop shared by every ``train_<model>.py``.

Every competition trainer repeats the same skeleton: ``StratifiedKFold`` with
``random_state=42`` (fold alignment across models is what makes OOF arrays
blendable), ``oof[val_idx] = val_pred``, ``test += test_pred / n_splits``.
:func:`run_cv` owns that skeleton; the caller keeps the model, metric,
saving, and submission logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def make_splitter(
    *, n_splits: int = 5, seed: int = 42, stratified: bool = True
) -> StratifiedKFold | KFold:
    """The splitter :func:`run_cv` uses (shuffled, ``random_state=seed``).

    Exposed so fold assignments can be reproduced exactly — see
    :func:`fold_assignment`.
    """
    splitter_cls = StratifiedKFold if stratified else KFold
    return splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)


def fold_assignment(
    y: NDArray[Any], *, n_splits: int = 5, seed: int = 42, stratified: bool = True
) -> NDArray[np.int64]:
    """1-based validation fold of every training row, exactly as :func:`run_cv`.

    Stored next to OOF predictions (``kaggle.predictions.save_predictions``)
    so a later reader can verify that two OOF arrays share their folds.

    >>> fold_assignment(np.array([0, 1] * 4), n_splits=2).tolist().count(1)
    4
    """
    folds = np.zeros(len(y), dtype=np.int64)
    splitter = make_splitter(n_splits=n_splits, seed=seed, stratified=stratified)
    for fold, (_, va_idx) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        folds[va_idx] = fold
    return folds


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

    splitter = make_splitter(n_splits=n_splits, seed=seed, stratified=stratified)
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


class BaggedCVResult(NamedTuple):
    """Seed-averaged OOF/test predictions plus the per-seed arrays behind them.

    ``per_seed_oof``/``per_seed_test`` have a leading seed axis. Keep them: the
    M-slope screen (:func:`data_science_stuff.kaggle.screening.screen_mechanism`)
    needs the paired delta at M = 1, 2, 3, 4 members, which is a prefix mean
    over this axis.
    """

    oof: NDArray[np.float64]
    test: NDArray[np.float64]
    fold_scores: list[float]
    per_seed_oof: NDArray[np.float64]
    per_seed_test: NDArray[np.float64]


def run_cv_seeds(
    X: Matrix,
    y: NDArray[Any],
    X_test: Matrix,
    make_fit_fold: Callable[[int], FitFoldFn],
    *,
    seeds: Sequence[int] = (42, 43, 44),
    resplit: bool = True,
    n_splits: int = 5,
    split_seed: int = 42,
    n_outputs: int = 1,
    stratified: bool = True,
    score_fn: ScoreFn | None = None,
    after_fold: Callable[[int], None] | None = None,
) -> BaggedCVResult:
    """Multi-seed bagged CV: :func:`run_cv` once per seed, then averaged.

    Extracted from s6e7's ``train_common.bagged_cv`` (148 importing scripts)
    and the ``for seed in SEEDS`` loops of s6e6/s6e8. Every row's averaged
    OOF is still out-of-fold for every seed, so the average is honest.

    Args:
        X: Training features, in ``build_features()`` order.
        y: Training target, same order.
        X_test: Test features.
        make_fit_fold: ``seed -> fit_fold``; the returned closure is used
            exactly like :func:`run_cv`'s ``fit_fold``, with the model seeded
            from ``seed``.
        seeds: Model seeds (one bagged member each).
        resplit: When True (s6e7's recipe) each seed also reshuffles the
            folds (splitter ``random_state=seed``), which decorrelates the
            members further. When False every seed uses ``split_seed`` —
            choose this when the per-seed OOF arrays must share folds with
            other models' OOF (e.g. for a fold-aligned stacker).
        n_splits: Folds per seed.
        split_seed: Splitter seed when ``resplit`` is False.
        n_outputs: As in :func:`run_cv`.
        stratified: As in :func:`run_cv`.
        score_fn: As in :func:`run_cv`; scores of all seed x fold fits are
            concatenated in ``fold_scores``.
        after_fold: As in :func:`run_cv` (put ``torch.cuda.empty_cache()``
            here for GPU models).

    Returns:
        :class:`BaggedCVResult`.

    >>> import numpy as np
    >>> X = np.arange(40, dtype=float).reshape(20, 2)
    >>> y = np.array([0, 1] * 10)
    >>> def make(seed):
    ...     return lambda xt, yt, xv, yv, xte, f: (
    ...         np.full(len(xv), yt.mean()), np.full(len(xte), yt.mean())
    ...     )
    >>> res = run_cv_seeds(X, y, X[:4], make, seeds=(1, 2), n_splits=2)
    >>> res.per_seed_oof.shape
    (2, 20)
    """
    if not seeds:
        msg = "seeds must be non-empty"
        raise ValueError(msg)
    results = [
        run_cv(
            X,
            y,
            X_test,
            make_fit_fold(seed),
            n_splits=n_splits,
            seed=seed if resplit else split_seed,
            n_outputs=n_outputs,
            stratified=stratified,
            score_fn=score_fn,
            after_fold=after_fold,
        )
        for seed in seeds
    ]
    per_seed_oof = np.stack([r.oof for r in results])
    per_seed_test = np.stack([r.test for r in results])
    fold_scores = [s for r in results for s in r.fold_scores]
    return BaggedCVResult(
        per_seed_oof.mean(axis=0),
        per_seed_test.mean(axis=0),
        fold_scores,
        per_seed_oof,
        per_seed_test,
    )
