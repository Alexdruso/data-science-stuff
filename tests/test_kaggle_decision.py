"""Tests for data_science_stuff.kaggle.decision."""

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

from data_science_stuff.kaggle.decision import (
    cascade_combine,
    cost_decide,
    fit_cost_matrix,
    make_cost_matrix,
    optimize_thresholds,
    split_half_gate,
)
from data_science_stuff.kaggle_utils import weighted_predict


def _imbalanced_proba() -> "tuple[NDArray[np.float64], NDArray[np.int64]]":
    """3-class problem where argmax suppresses the rare class (recall 0)."""
    y = np.array([0] * 90 + [1] * 90 + [2] * 20, dtype=np.int64)
    proba = np.empty((len(y), 3), dtype=np.float64)
    proba[y == 0] = [0.70, 0.20, 0.10]
    proba[y == 1] = [0.20, 0.70, 0.10]
    proba[y == 2] = [0.46, 0.15, 0.39]  # argmax sends every rare row to class 0
    return proba, y


def test_optimize_thresholds_beats_argmax_and_reports_true_score() -> None:
    proba, y = _imbalanced_proba()
    base = float(balanced_accuracy_score(y, np.argmax(proba, axis=1)))

    weights, score = optimize_thresholds(proba, y, n_restarts=4)

    assert score > base
    assert weights.shape == (3,)
    assert weights.max() == pytest.approx(1.0)  # max-normalized
    achieved = float(balanced_accuracy_score(y, weighted_predict(proba, weights)))
    assert score == pytest.approx(achieved)


def test_make_cost_matrix_shapes_and_fill_order() -> None:
    m3 = make_cost_matrix(np.zeros(6), 3)
    assert m3.shape == (3, 3)
    assert np.trace(m3) == 0.0

    m4 = make_cost_matrix(np.log(np.arange(1, 13, dtype=float)), 4)
    assert m4.shape == (4, 4)
    assert m4[0, 1] == pytest.approx(1.0)  # first off-diagonal, row-major
    assert m4[3, 2] == pytest.approx(12.0)  # last off-diagonal


def test_make_cost_matrix_validates_length() -> None:
    with pytest.raises(ValueError, match="off-diagonal"):
        make_cost_matrix(np.zeros(5), 3)


def test_cost_decide_uniform_costs_reduce_to_argmax() -> None:
    proba, _ = _imbalanced_proba()
    pred = cost_decide(proba, make_cost_matrix(np.zeros(6), 3))
    np.testing.assert_array_equal(pred, np.argmax(proba, axis=1))


def test_fit_cost_matrix_warm_start_never_below_per_class() -> None:
    proba, y = _imbalanced_proba()
    weights, per_class_score = optimize_thresholds(proba, y, n_restarts=4)

    log_costs, cost_score = fit_cost_matrix(proba, y, weights, n_restarts=2)

    assert log_costs.shape == (6,)
    assert cost_score >= per_class_score - 1e-12


def test_split_half_gate_returns_two_deltas_per_seed() -> None:
    proba, y = _imbalanced_proba()
    deltas = split_half_gate(proba, y, n_seeds=1, n_restarts=1)
    assert len(deltas) == 2
    assert all(isinstance(d, float) for d in deltas)


def test_cascade_combine_matches_manual_factorization() -> None:
    rng = np.random.default_rng(1)
    p_qso = rng.random(50)
    p_star_cond = rng.random(50)

    out = cascade_combine([p_qso, p_star_cond], [1, 2, 0])

    np.testing.assert_allclose(out[:, 1], p_qso)
    np.testing.assert_allclose(out[:, 2], (1 - p_qso) * p_star_cond)
    np.testing.assert_allclose(out[:, 0], (1 - p_qso) * (1 - p_star_cond))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_cascade_combine_validates_lengths() -> None:
    with pytest.raises(ValueError, match="one more entry"):
        cascade_combine([np.array([0.5])], [0, 1, 2])
