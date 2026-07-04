"""Tests for data_science_stuff.kaggle.stacking."""

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from data_science_stuff.kaggle.stacking import (
    caruana_select,
    clipped_logit,
    stack_oof,
)


def _meta() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)


def _three_class_bases(
    n: int = 240,
) -> "tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], NDArray[np.int64]]":
    """Two noisy-but-informative base models over a 3-class target."""
    rng = np.random.default_rng(0)
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3)

    def noisy_oof(noise: float, seed: int) -> "NDArray[np.float64]":
        local = np.random.default_rng(seed)
        proba: NDArray[np.float64] = np.full((n, 3), 0.1)
        proba[np.arange(n), y] = 0.8
        proba += local.normal(0, noise, proba.shape)
        proba = np.clip(proba, 1e-3, None)
        return np.asarray(proba / proba.sum(axis=1, keepdims=True), dtype=np.float64)

    oofs = [noisy_oof(0.15, 1), noisy_oof(0.15, 2)]
    tests = [rng.dirichlet(np.ones(3), size=30) for _ in oofs]
    return oofs, tests, y


def test_clipped_logit_clips_extremes() -> None:
    p = np.array([0.0, 0.5, 1.0])
    z = clipped_logit(p, eps=1e-6)
    assert np.isfinite(z).all()
    assert z[0] == pytest.approx(np.log(1e-6))
    assert z[1] == pytest.approx(np.log(0.5))


def test_stack_oof_shapes_simplex_and_signal() -> None:
    oofs, tests, y = _three_class_bases()

    stack_oof_arr, stack_test = stack_oof(
        oofs, tests, y, _meta, seeds=(0, 1), n_splits=3
    )

    assert stack_oof_arr.shape == (len(y), 3)
    assert stack_test.shape == (30, 3)
    np.testing.assert_allclose(stack_oof_arr.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(stack_test.sum(axis=1), 1.0, atol=1e-9)
    # The stack extracts the signal the bases carry.
    stacked = balanced_accuracy_score(y, np.argmax(stack_oof_arr, axis=1))
    assert stacked > 0.9


def test_stack_oof_is_honest_versus_in_sample_fit() -> None:
    oofs, tests, y = _three_class_bases()

    stack_oof_arr, _ = stack_oof(oofs, tests, y, _meta, seeds=(0,), n_splits=3)
    honest = balanced_accuracy_score(y, np.argmax(stack_oof_arr, axis=1))

    # Same meta fit AND scored on all rows — the leaky version.
    z = np.hstack([clipped_logit(a) for a in oofs])
    leaky_meta = _meta().fit(z, y)
    leaky = balanced_accuracy_score(y, leaky_meta.predict(z))
    assert honest <= leaky + 1e-9


def test_stack_oof_binary_one_dimensional_bases() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60, dtype=np.int64)
    base = np.clip(y * 0.8 + rng.normal(0.1, 0.1, len(y)), 0.001, 0.999)
    base_test = rng.random(15)

    stack_oof_arr, stack_test = stack_oof(
        [base, base], [base_test, base_test], y, _meta, seeds=(0,), n_splits=3
    )

    assert stack_oof_arr.shape == (len(y), 2)
    assert stack_test.shape == (15, 2)


def test_stack_oof_validates_inputs() -> None:
    y = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="same models"):
        stack_oof([np.zeros((2, 2))], [], y, _meta)
    with pytest.raises(ValueError, match="at least one"):
        stack_oof([], [], y, _meta)


def test_caruana_select_prefers_complementary_pair() -> None:
    n = 120
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3)
    rng = np.random.default_rng(0)

    def specialist(strong_class: int) -> "NDArray[np.float64]":
        proba = np.full((n, 3), 1 / 3)
        strong_rows = y == strong_class
        proba[strong_rows] = 0.05
        proba[strong_rows, strong_class] = 0.9
        # Slight signal elsewhere so singles beat noise.
        other = ~strong_rows
        proba[other, y[other]] += 0.05
        return np.asarray(proba / proba.sum(axis=1, keepdims=True), dtype=np.float64)

    noise = rng.dirichlet(np.ones(3), size=n)
    library = np.stack([specialist(0), specialist(1), noise])

    result = caruana_select(library, y, n_steps=8, init_topk=1, n_outer=3)

    assert len(result.holdout_scores) == 3
    assert len(result.full_picks) == 8
    # The greedy composition leans on the two specialists, not the noise model.
    assert {0, 1}.issubset(set(result.full_picks))
    picks = np.array(result.full_picks)
    assert (picks == 2).sum() < len(picks) / 2

    def single_score(m: int) -> float:
        return float(balanced_accuracy_score(y, np.argmax(library[m], axis=1)))

    assert result.full_score >= max(single_score(m) for m in range(3))
