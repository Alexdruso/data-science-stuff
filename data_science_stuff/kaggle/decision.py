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
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict


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
