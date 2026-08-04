"""Decision-rule post-processing: per-class weights, cost matrices, cascades.

The cheapest honest lift on imbalanced multiclass problems: keep the model's
probabilities and optimize the *decision rule* on OOF. In increasing order of
expressiveness:

1. :func:`optimize_thresholds` — per-class multiplicative weights,
   ``argmax(proba * w)``. n_classes params; the s6e6 workhorse (+0.0013).
2. :func:`fit_cost_matrix` — full Bayes cost matrix,
   ``argmin(proba @ C)``. n²-n params; strictly generalizes (1), so always
   gate it honestly with :func:`split_half_gate` before trusting it.

:func:`cascade_combine` recombines nested-binary cascade stages (a diversity
lever: factor a k-class problem into sequential binaries).

**Which per-class weight fitter to use.** ``argmax(proba * w)`` scored by
balanced accuracy is a *piecewise-constant* objective, and Nelder-Mead cannot
reach a tight tolerance on a step function — it burns its whole iteration
budget on every call (s6e7 measured ~45k objective evaluations per call, which
turned a 5-minute probe into a multi-hour one). :func:`tune_decision_weights_grid`
runs a coarse-to-fine grid instead: ~30x faster *and* it found a strictly
better optimum (+0.000077 balanced accuracy on a deployed 8-seed ensemble).
Prefer it for new work and inside any loop;
:func:`data_science_stuff.kaggle_utils.tune_decision_weights` (Nelder-Mead)
stays the default of :func:`optimize_thresholds` so previously scored
artifacts remain reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

# Fits per-class weights from ``(y, proba)``; both weight fitters match it.
WeightFitter = Callable[[NDArray[np.int64], NDArray[np.float64]], NDArray[np.float64]]

# Class count with a dedicated log-ratio fast path in the grid weight fitter.
FAST_PATH_CLASSES = 3


def optimize_thresholds(
    oof_proba: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    n_restarts: int = 15,
    seed: int = 42,
) -> tuple[NDArray[np.float64], float]:
    """Per-class decision weights plus the achieved balanced accuracy.

    Thin wrapper over
    :func:`data_science_stuff.kaggle_utils.tune_decision_weights` returning
    the ``(weights, score)`` pair used throughout the competition scripts.
    Apply via ``argmax(proba * weights)`` (:func:`weighted_predict`), with the
    identical weights on test predictions. Weights are max-normalized (the
    decision rule is scale-invariant, so this changes nothing observable).

    Args:
        oof_proba: (n, n_classes) out-of-fold probabilities.
        y: Encoded ground-truth labels.
        n_restarts: Nelder-Mead random restarts.
        seed: RNG seed for the restart initial points.

    Returns:
        ``(weights, score)``: the (n_classes,) weight vector and the balanced
        accuracy it achieves on ``oof_proba``.
    """
    weights = tune_decision_weights(y, oof_proba, seed=seed, n_restarts=n_restarts)
    score = float(balanced_accuracy_score(y, weighted_predict(oof_proba, weights)))
    return weights, score


def balanced_accuracy_fast(
    y: NDArray[np.int64],
    pred: NDArray[np.int64],
    *,
    n_classes: int | None = None,
) -> float:
    """Balanced accuracy (mean per-class recall) via two ``bincount`` calls.

    Equivalent to :func:`sklearn.metrics.balanced_accuracy_score` (classes
    absent from ``y`` are skipped, as sklearn does) but without building a
    confusion matrix — worth having in any loop that scores millions of rows
    thousands of times, such as :mod:`data_science_stuff.kaggle.screening`.

    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> balanced_accuracy_fast(y, np.array([0, 0, 1, 0]))
    0.75
    """
    k = n_classes if n_classes is not None else int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(np.float64)
    hits = np.bincount(y[pred == y], minlength=len(counts)).astype(np.float64)
    present = counts > 0
    return float(np.mean(hits[present] / counts[present]))


def _grid_weights_3class(
    y: NDArray[np.int64],
    proba: NDArray[np.float64],
    grid: int,
    rounds: int,
    span: float,
) -> NDArray[np.float64]:
    """Exhaustive coarse-to-fine 2-D grid over the two free log-weights.

    The decision depends only on the two log-ratios: with weights ``(1, a, b)``
    and ``u = log p1 - log p0``, ``v = log p2 - log p0``, class 1 wins iff
    ``u + log a > 0`` and ``u + log a >= v + log b``. Precomputing ``u, v``
    makes each grid point two adds and three comparisons rather than an
    ``(n, 3)`` multiply plus argmax.
    """
    counts = np.bincount(y, minlength=FAST_PATH_CLASSES).astype(np.float64)
    log_p = np.log(np.clip(proba, 1e-12, None)).astype(np.float32)
    u = log_p[:, 1] - log_p[:, 0]
    v = log_p[:, 2] - log_p[:, 0]
    is_0, is_1, is_2 = (y == c for c in range(FAST_PATH_CLASSES))

    def score_at(log_a: float, log_b: float) -> float:
        a, b = u + np.float32(log_a), v + np.float32(log_b)
        one = (a > 0) & (a >= b)
        two = (b > 0) & (b > a)
        zero = ~(one | two)
        return float(
            (
                np.count_nonzero(zero & is_0) / counts[0]
                + np.count_nonzero(one & is_1) / counts[1]
                + np.count_nonzero(two & is_2) / counts[2]
            )
            / 3.0,
        )

    center = np.zeros(2)
    for _ in range(rounds):
        axes = [np.linspace(c - span / 2, c + span / 2, grid) for c in center]
        best, best_point = -1.0, center
        for log_a in axes[0]:
            for log_b in axes[1]:
                value = score_at(float(log_a), float(log_b))
                if value > best:
                    best, best_point = value, np.array([log_a, log_b])
        center, span = best_point, span / (grid / 3.0)
    return np.concatenate([[0.0], center])


def _grid_weights_general(
    y: NDArray[np.int64],
    proba: NDArray[np.float64],
    grid: int,
    rounds: int,
    span: float,
) -> NDArray[np.float64]:
    """Coarse-to-fine *coordinate* grid over the K-1 free log-weights.

    A full K-1 dimensional grid costs ``grid**(K-1)`` evaluations per round;
    sweeping one coordinate at a time costs ``grid * (K - 1)`` and is adequate
    here because the surface is nearly separable in the class weights (s6e7
    measured weight vectors 0.2 apart scoring within 2e-5 of each other).
    """
    n_classes = proba.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    log_p = np.log(np.clip(proba, 1e-12, None))
    log_w = np.zeros(n_classes)

    def score_at(candidate: NDArray[np.float64]) -> float:
        pred = np.asarray(np.argmax(log_p + candidate, axis=1), dtype=np.int64)
        hits = np.bincount(y[pred == y], minlength=n_classes).astype(np.float64)
        present = counts > 0
        return float(np.mean(hits[present] / counts[present]))

    best = score_at(log_w)
    for _ in range(rounds):
        for j in range(1, n_classes):
            axis = np.linspace(log_w[j] - span / 2, log_w[j] + span / 2, grid)
            for value in axis:
                candidate = log_w.copy()
                candidate[j] = value
                score = score_at(candidate)
                if score > best:
                    best, log_w = score, candidate
        span /= grid / 3.0
    return log_w


def tune_decision_weights_grid(
    y: NDArray[np.int64],
    proba: NDArray[np.float64],
    *,
    grid: int = 17,
    rounds: int = 3,
    span: float = 2.4,
    fit_n: int = 0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Per-class decision weights by coarse-to-fine grid search (max-normalized).

    A grid is the right tool for the piecewise-constant surface
    ``balanced_accuracy(argmax(proba * w))``: it cannot be fooled by the flat
    regions that make a simplex method exhaust its iteration budget without
    converging. Drop-in alternative to
    :func:`data_science_stuff.kaggle_utils.tune_decision_weights` — see the
    module docstring for why it is both faster and (measurably) better.

    Class 0 is pinned at weight 1 (the rule is scale-invariant, so there are
    K-1 free parameters). With ``K == 3`` an exhaustive 2-D grid over the two
    log-ratios is used; otherwise coordinate sweeps.

    Args:
        y: Encoded ground-truth labels.
        proba: (n, n_classes) probabilities.
        grid: Points per axis per round.
        rounds: Coarse-to-fine refinement rounds (each shrinks the span).
        span: Width of the initial log-weight window.
        fit_n: If non-zero, subsample this many rows to *fit* the weights. The
            weight vector has only K-1 free parameters, so a large fit sample
            buys nothing; score on the full split via
            :func:`data_science_stuff.kaggle_utils.weighted_predict`.
        seed: RNG seed for the ``fit_n`` subsample.

    Returns:
        (n_classes,) weights normalized so the largest entry is 1.
    """
    if fit_n and len(y) > fit_n:
        idx = np.random.default_rng(seed).choice(len(y), fit_n, replace=False)
        y, proba = y[idx], proba[idx]
    if proba.shape[1] == FAST_PATH_CLASSES:
        log_w = _grid_weights_3class(y, proba, grid, rounds, span)
    else:
        log_w = _grid_weights_general(y, proba, grid, rounds, span)
    weights = np.exp(log_w)
    return np.asarray(weights / weights.max(), dtype=np.float64)


def bagged_decision_weights(
    y: NDArray[np.int64],
    proba: NDArray[np.float64],
    *,
    n_bags: int = 8,
    seed: int = 42,
    fitter: WeightFitter | None = None,
) -> NDArray[np.float64]:
    """Bootstrap-bagged per-class weights, averaged in log space.

    A single fit on the full OOF is unstable: s6e7 measured ~0.004 balanced
    accuracy of spread depending on the slice the optimizer saw. Averaging the
    log-weights over bootstrap resamples costs ``n_bags`` fits and removes that
    dependence — cheap now that the default fitter is the grid rather than
    Nelder-Mead.

    Args:
        y: Encoded ground-truth labels.
        proba: (n, n_classes) out-of-fold probabilities.
        n_bags: Number of bootstrap resamples.
        seed: RNG seed for the resample indices.
        fitter: Weight fitter, default :func:`tune_decision_weights_grid`. Pass
            :func:`data_science_stuff.kaggle_utils.tune_decision_weights` to
            reproduce Nelder-Mead-era artifacts.

    Returns:
        (n_classes,) weights normalized so the largest entry is 1.
    """
    fit = tune_decision_weights_grid if fitter is None else fitter
    rng = np.random.default_rng(seed)
    log_weights = []
    for _ in range(n_bags):
        idx = rng.integers(0, len(y), len(y))
        log_weights.append(np.log(np.asarray(fit(y[idx], proba[idx]))))
    weights = np.exp(np.mean(log_weights, axis=0))
    return np.asarray(weights / weights.max(), dtype=np.float64)


def _off_diagonal(n_classes: int) -> list[tuple[int, int]]:
    """(j, k) index pairs with j != k: cost of predicting k when truth is j."""
    return [(j, k) for j in range(n_classes) for k in range(n_classes) if j != k]


def make_cost_matrix(
    log_costs: NDArray[np.float64], n_classes: int
) -> NDArray[np.float64]:
    """Build an (n, n) cost matrix from its n²-n off-diagonal log-costs.

    The diagonal is zero (correct predictions cost nothing); off-diagonal
    entries are filled row-major with ``exp(log_costs)`` so any real-valued
    optimizer input yields positive costs.

    >>> import numpy as np
    >>> make_cost_matrix(np.zeros(6), 3).tolist()
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    """
    pairs = _off_diagonal(n_classes)
    if len(log_costs) != len(pairs):
        msg = f"expected {len(pairs)} off-diagonal log-costs, got {len(log_costs)}"
        raise ValueError(msg)
    cost = np.zeros((n_classes, n_classes))
    for (j, k), value in zip(pairs, np.exp(np.asarray(log_costs, dtype=np.float64))):
        cost[j, k] = value
    return cost


def cost_decide(
    proba: NDArray[np.float64], cost: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Bayes decision rule: ``argmin_k`` of expected cost ``sum_j P(j|x) C[j, k]``.

    With all off-diagonal costs equal this reduces to plain argmax.
    """
    return np.asarray(np.argmin(proba @ cost, axis=1), dtype=np.int64)


def fit_cost_matrix(
    proba: NDArray[np.float64],
    y: NDArray[np.int64],
    warm_weights: NDArray[np.float64],
    *,
    n_restarts: int = 8,
    maxiter: int = 2000,
    xatol: float = 1e-7,
    fatol: float = 1e-7,
) -> tuple[NDArray[np.float64], float]:
    """Nelder-Mead over the off-diagonal log-costs, warm-started.

    Per-class scaling ``argmax(w_k P_k)`` equals the cost matrix
    ``C[j, k] = w_j`` (k != j), so warm-starting from the per-class optimum
    guarantees the in-sample optimum is at least the per-class one. A pair
    of classes can then be penalized *specifically* (e.g. one confusion pair
    dominating the errors), which no per-class scale can express.

    Args:
        proba: (n, n_classes) probabilities.
        y: Encoded ground-truth labels.
        warm_weights: Per-class weights from :func:`optimize_thresholds`.
        n_restarts: Perturbed restarts around the warm start (seeded 0..n-1).
        maxiter: Nelder-Mead iteration cap.
        xatol: Nelder-Mead absolute tolerance on the simplex.
        fatol: Nelder-Mead absolute tolerance on the objective.

    Returns:
        ``(log_costs, score)``: pass ``log_costs`` to :func:`make_cost_matrix`.
    """
    n_classes = proba.shape[1]
    pairs = _off_diagonal(n_classes)

    def neg_score(log_c: NDArray[np.float64]) -> float:
        pred = cost_decide(proba, make_cost_matrix(log_c, n_classes))
        return -float(balanced_accuracy_score(y, pred))

    warm = np.array([np.log(warm_weights[j]) for j, _ in pairs])
    best_x, best_score = warm, -neg_score(warm)
    starts = [warm] + [
        warm + np.random.default_rng(seed).normal(0, 0.3, size=len(pairs))
        for seed in range(n_restarts)
    ]
    for x0 in starts:
        result = minimize(
            neg_score,
            x0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol},
        )
        if -float(result.fun) > best_score:
            best_score = -float(result.fun)
            best_x = np.asarray(result.x, dtype=np.float64)
    return best_x, best_score


def split_half_gate(
    proba: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    n_seeds: int = 3,
    n_restarts: int = 5,
) -> list[float]:
    """Honest holdout deltas (cost-matrix - per-class) over seed/half swaps.

    Both decision rules are fit on one half of the OOF rows and scored on the
    other, for ``n_seeds`` random splits with both half assignments. Gate a
    cost-matrix submission on ``np.mean(result) >= 0`` — the extra n²-n
    parameters must generalize, not just fit in-sample.

    Returns:
        ``2 * n_seeds`` holdout deltas (cost-matrix score - per-class score).
    """
    n_classes = proba.shape[1]
    deltas: list[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        halves = (idx[: len(y) // 2], idx[len(y) // 2 :])
        for fit_idx, val_idx in (halves, halves[::-1]):
            w, _ = optimize_thresholds(
                proba[fit_idx], y[fit_idx], n_restarts=n_restarts
            )
            per_class = float(
                balanced_accuracy_score(y[val_idx], weighted_predict(proba[val_idx], w))
            )
            log_c, _ = fit_cost_matrix(
                proba[fit_idx], y[fit_idx], w, n_restarts=n_restarts
            )
            cost_mx = float(
                balanced_accuracy_score(
                    y[val_idx],
                    cost_decide(proba[val_idx], make_cost_matrix(log_c, n_classes)),
                )
            )
            deltas.append(cost_mx - per_class)
    return deltas


def cascade_combine(
    stage_probs: list[NDArray[np.float64]],
    class_order: list[int],
) -> NDArray[np.float64]:
    """Recombine nested-binary cascade stages into full class probabilities.

    A cascade factors ``P(class | x)`` into sequential binaries:
    ``stage_probs[i]`` is ``P(class_order[i] | not any earlier class)``. The
    last class in ``class_order`` receives the remaining mass, so
    ``len(class_order) == len(stage_probs) + 1`` and every row sums to 1.

    Example (s6e6): ``cascade_combine([p_qso, p_star_cond], [QSO, STAR, GAL])``
    yields ``P(GAL) = (1-p_qso)(1-p_star)``, ``P(QSO) = p_qso``,
    ``P(STAR) = (1-p_qso) p_star``.

    >>> import numpy as np
    >>> cascade_combine([np.array([0.5]), np.array([0.4])], [1, 2, 0]).round(2).tolist()
    [[0.3, 0.5, 0.2]]
    """
    if len(class_order) != len(stage_probs) + 1:
        msg = "class_order must have exactly one more entry than stage_probs"
        raise ValueError(msg)
    n_rows = len(stage_probs[0])
    out = np.zeros((n_rows, len(class_order)))
    rest: NDArray[np.float64] = np.ones(n_rows)
    for cls, p in zip(class_order, stage_probs):
        p_arr = np.asarray(p, dtype=np.float64)
        out[:, cls] = rest * p_arr
        rest = rest * (1.0 - p_arr)
    out[:, class_order[-1]] = rest
    return out
