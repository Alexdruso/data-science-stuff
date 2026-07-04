"""Stacking: honest multi-seed OOF meta-learning and Caruana greedy selection.

A scalar blend (see :mod:`data_science_stuff.kaggle.blending`) assigns one
weight per model, so a model that is strong on one class and weak on another
gets ~0 weight — no scalar captures "use its class-A column, ignore its
class-B column". :func:`stack_oof` fits a meta-learner on the base models'
per-class outputs instead (a weight per model-class pair with a linear meta),
which is what broke s6e6's 0.9662 plateau (LR-on-logits stack → 0.9705).

Honesty rules baked in:

- The meta-learner is fit on K-1 folds of the OOF rows and predicts the
  held-out fold — never fit and scored on the same rows.
- :func:`caruana_select` reports held-out scores from an outer CV around the
  greedy selection; naive best-on-OOF selection manufactures a fake gain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from data_science_stuff.kaggle.blending import ScoreFn


def clipped_logit(p: NDArray[np.float64], *, eps: float = 1e-6) -> NDArray[np.float64]:
    """Elementwise ``log(p)`` with probabilities clipped away from {0, 1}.

    Stacking on logits instead of raw probabilities lets a linear meta-learner
    operate on an unbounded, roughly Gaussian scale.

    >>> import numpy as np
    >>> clipped_logit(np.array([0.5])).round(4).tolist()
    [-0.6931]
    """
    return np.asarray(np.log(np.clip(p, eps, 1 - eps)), dtype=np.float64)


def _as_columns(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def stack_oof(
    base_oofs: Sequence[NDArray[np.float64]],
    base_tests: Sequence[NDArray[np.float64]],
    y: NDArray[np.int64],
    meta_factory: Callable[[], Any],
    *,
    seeds: Sequence[int] = (2024, 7, 13, 42, 99),
    n_splits: int = 5,
    use_logits: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Honest multi-seed OOF stacking over base-model prediction arrays.

    The feature matrix is the horizontal stack of every base model's OOF
    array (optionally logit-transformed). For each seed, a fresh meta-learner
    is fit on K-1 folds of those rows and predicts the held-out fold; test
    predictions are averaged over every (seed, fold) fit. Multi-seed
    averaging smooths the fold-boundary noise of any single split.

    Prune the base list to strong, *diverse* models first: near-duplicate
    variants add only collinearity (on s6e6, six weak bases added +0.0001).
    If the stack is flat, the missing ingredient is a diverse base — a
    different feature space, loss, or decomposition — not a better meta.

    Args:
        base_oofs: One OOF array per base model, aligned with ``y``
            ((n, n_classes) probabilities, or 1-D for binary/regression).
        base_tests: Matching test arrays, same order and widths.
        y: Encoded training labels (drives the stratified folds).
        meta_factory: Zero-arg callable returning a *fresh* estimator with
            ``fit`` / ``predict_proba`` (e.g. ``lambda: LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000)``).
        seeds: One outer CV per seed; predictions averaged across them.
        n_splits: Folds per seed.
        use_logits: Feed :func:`clipped_logit` of the arrays to the meta
            (recommended for linear metas; use False for tree metas).

    Returns:
        ``(oof, test)`` stacked probability arrays, shape (n, n_classes) /
        (n_test, n_classes).
    """
    if len(base_oofs) != len(base_tests):
        msg = "base_oofs and base_tests must list the same models"
        raise ValueError(msg)
    if not base_oofs:
        msg = "need at least one base model to stack"
        raise ValueError(msg)

    transform = clipped_logit if use_logits else _as_columns
    z_train = np.hstack([transform(_as_columns(a)) for a in base_oofs])
    z_test = np.hstack([transform(_as_columns(a)) for a in base_tests])

    n_classes = int(np.unique(y).size)
    oof = np.zeros((len(y), n_classes))
    test = np.zeros((len(z_test), n_classes))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(z_train, y):
            meta = meta_factory()
            meta.fit(z_train[tr_idx], y[tr_idx])
            oof[va_idx] += meta.predict_proba(z_train[va_idx]) / len(seeds)
            test += meta.predict_proba(z_test) / (len(seeds) * n_splits)
    return oof, test


class CaruanaResult(NamedTuple):
    """Held-out scores per outer fold, plus the full-data greedy composition."""

    holdout_scores: list[float]
    full_picks: list[int]
    full_score: float


def _greedy_select(
    library: NDArray[np.float64],
    y: NDArray[np.int64],
    idx: NDArray[Any],
    *,
    n_steps: int,
    init_topk: int,
    score: ScoreFn,
) -> list[int]:
    """Greedy forward selection with replacement on the given rows."""
    n_models = library.shape[0]
    singles = [float(score(y[idx], library[m][idx])) for m in range(n_models)]
    picks = [int(m) for m in np.argsort(singles)[::-1][:init_topk]]
    running = library[picks][:, idx, :].sum(axis=0)
    for _ in range(n_steps - len(picks)):
        best_m, best_score = -1, -np.inf
        for m in range(n_models):
            candidate_score = float(score(y[idx], running + library[m][idx]))
            if candidate_score > best_score:
                best_score, best_m = candidate_score, m
        picks.append(best_m)
        running = running + library[best_m][idx]
    return picks


def caruana_select(
    oofs: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    n_steps: int = 50,
    init_topk: int = 3,
    n_outer: int = 5,
    seed: int = 42,
    score_fn: ScoreFn | None = None,
) -> CaruanaResult:
    """Caruana greedy forward selection (with replacement), honestly gated.

    Averages probabilities of greedily-picked base models (a model picked
    twice counts twice). Selection runs inside an outer CV so
    ``holdout_scores`` is an honest estimate — compare its mean against the
    incumbent combiner before trusting the full-data composition. Exclude
    meta/stack arrays from ``oofs`` (selecting a stack is circular).

    Args:
        oofs: (n_models, n_rows, n_classes) library of base OOF arrays.
        y: Encoded training labels.
        n_steps: Total greedy picks (with replacement).
        init_topk: Seed the ensemble with the K best singletons.
        n_outer: Outer honest-CV folds.
        seed: Outer CV ``random_state``.
        score_fn: ``(y_true, summed_probs) -> float``; defaults to balanced
            accuracy on argmax.

    Returns:
        :class:`CaruanaResult` — held-out fold scores, the (optimistic)
        full-data pick list, and its in-sample score.
    """
    score: ScoreFn = (
        score_fn
        if score_fn is not None
        else lambda yt, p: float(balanced_accuracy_score(yt, np.argmax(p, axis=1)))
    )
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    holdout_scores: list[float] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        picks = _greedy_select(
            oofs, y, tr_idx, n_steps=n_steps, init_topk=init_topk, score=score
        )
        held_out = oofs[picks][:, va_idx, :].sum(axis=0)
        holdout_scores.append(float(score(y[va_idx], held_out)))

    full_picks = _greedy_select(
        oofs, y, np.arange(len(y)), n_steps=n_steps, init_topk=init_topk, score=score
    )
    full_score = float(score(y, oofs[full_picks].sum(axis=0)))
    return CaruanaResult(holdout_scores, full_picks, full_score)
