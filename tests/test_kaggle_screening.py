"""Tests for data_science_stuff.kaggle.screening (the M-slope instrument)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from data_science_stuff.kaggle.screening import (
    INFORMATION,
    NOT_INFORMATION,
    SELECTION_NOISE,
    SLOW_DECAY,
    UNDEFINED,
    MechanismScreen,
    _subsets,
    argmax_layer,
    calibrate_lambda,
    classify_exponent,
    decay_exponent,
    frozen_weight_layer,
    honest_weighted_layer,
    mean_shift_control,
    screen_mechanism,
    selection_null_arms,
    variance_reduction_control,
)


def _members(
    n_members: int = 8, n_rows: int = 4000, seed: int = 0
) -> "tuple[NDArray[np.float64], NDArray[np.int64]]":
    """A leg's per-member OOF probabilities.

    Three noise sources, matching the regime the instrument is used in: class
    signal, irreducible per-row difficulty shared by every member (so the
    ensemble does not saturate), and per-member noise (so averaging has
    something to remove).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(rng.integers(0, 3, n_rows), dtype=np.int64)
    base = 1.6 * np.eye(3)[y] + 1.2 * rng.normal(size=(n_rows, 3))
    members = np.empty((n_members, n_rows, 3))
    for i in range(n_members):
        logits = base + 1.3 * rng.normal(size=(n_rows, 3))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        members[i] = exp / exp.sum(axis=1, keepdims=True)
    return members, y


def test_decay_exponent_recovers_a_power_law() -> None:
    sizes = np.array([1, 2, 3, 4])
    assert decay_exponent(sizes, 0.01 / sizes) == pytest.approx(-1.0)
    assert decay_exponent(sizes, 0.01 / np.sqrt(sizes)) == pytest.approx(-0.5)
    assert decay_exponent(sizes, np.full(4, 0.01)) == pytest.approx(0.0, abs=1e-12)


def test_decay_exponent_is_undefined_when_the_delta_crosses_zero() -> None:
    assert np.isnan(decay_exponent([1, 2, 3], [0.01, -0.002, 0.001]))


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [
        (0.558, INFORMATION),
        (-0.1, SLOW_DECAY),
        (-0.5, SELECTION_NOISE),
        (-1.002, NOT_INFORMATION),
        (float("nan"), UNDEFINED),
    ],
)
def test_classify_exponent_names_the_nearest_anchor(
    exponent: float, expected: str
) -> None:
    assert classify_exponent(exponent) == expected


def test_classify_exponent_honours_a_measured_null_band() -> None:
    # -0.1 is "slow decay" against the anchors, but noise if the band says so.
    assert classify_exponent(-0.1) == SLOW_DECAY
    assert classify_exponent(-0.1, null_band=(-0.77, -0.06)) == SELECTION_NOISE


def test_subsets_enumerates_exhaustively_then_samples_under_the_cap() -> None:
    assert _subsets(4, 2, max_subsets=70, seed=0) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    sampled = _subsets(8, 4, max_subsets=5, seed=0)
    assert len(sampled) == 5
    assert len(set(sampled)) == 5
    assert all(len(s) == 4 for s in sampled)


def test_screen_mechanism_reads_a_mean_shift_at_the_information_end() -> None:
    """A shift toward the truth survives averaging — and here it grows under it."""
    pool, y = _members()
    arm = pool[:4]
    challenger = mean_shift_control(arm, y, lam=0.05)

    result = screen_mechanism(
        challenger, arm, y, layer=argmax_layer, n_splits=3, n_bootstrap=20
    )

    assert isinstance(result, MechanismScreen)
    assert result.sizes.tolist() == [1, 2, 3, 4]
    assert result.deltas[0] > 0
    assert result.exponent > 0
    assert result.verdict == INFORMATION
    assert "EXPONENT" in str(result)


def test_screen_mechanism_reads_variance_reduction_at_the_bottom() -> None:
    """Shrinking toward a disjoint member average decays roughly as 1/M."""
    pool, y = _members()
    arm, target = pool[:4], pool[4:]
    challenger = variance_reduction_control(arm, target, lam=0.7)

    result = screen_mechanism(
        challenger, arm, y, layer=argmax_layer, n_splits=3, n_bootstrap=20
    )

    assert result.deltas[0] > 0  # positive at M=1, which is what fools a level read
    assert result.exponent < -0.75
    assert result.verdict == NOT_INFORMATION
    assert result.exponent_sd >= 0.0


def test_screen_mechanism_separates_the_two_mechanisms() -> None:
    """The instrument's whole point: the two controls must not overlap."""
    pool, y = _members()
    arm, target = pool[:4], pool[4:]
    info = screen_mechanism(
        mean_shift_control(arm, y, lam=0.05),
        arm,
        y,
        layer=argmax_layer,
        n_splits=3,
        n_bootstrap=10,
    )
    noise = screen_mechanism(
        variance_reduction_control(arm, target, lam=0.7),
        arm,
        y,
        layer=argmax_layer,
        n_splits=3,
        n_bootstrap=10,
    )
    assert info.exponent - noise.exponent > 1.0


def test_screen_mechanism_honest_layer_runs_and_respects_max_size() -> None:
    pool, y = _members(n_members=3, n_rows=600)
    challenger = mean_shift_control(pool, y, lam=0.05)

    result = screen_mechanism(
        challenger,
        pool,
        y,
        max_size=2,
        layer=honest_weighted_layer,
        n_splits=1,
        n_bootstrap=5,
    )

    assert result.sizes.tolist() == [1, 2]
    assert result.deltas.shape == (2,)


def test_screen_mechanism_parallel_matches_the_in_process_run() -> None:
    pool, y = _members(n_members=3, n_rows=800)
    challenger = mean_shift_control(pool, y, lam=0.05)
    kwargs = {"layer": argmax_layer, "n_splits": 1, "n_bootstrap": 5}

    serial = screen_mechanism(challenger, pool, y, n_jobs=1, **kwargs)  # type: ignore[arg-type]
    parallel = screen_mechanism(challenger, pool, y, n_jobs=2, **kwargs)  # type: ignore[arg-type]

    np.testing.assert_allclose(serial.deltas, parallel.deltas)
    assert serial.exponent == pytest.approx(parallel.exponent)


def test_screen_mechanism_rejects_unpaired_or_misaligned_arms() -> None:
    members, y = _members(n_members=3, n_rows=120)
    with pytest.raises(ValueError, match="arm shapes differ"):
        screen_mechanism(members[:2], members, y)
    with pytest.raises(ValueError, match="row mismatch"):
        screen_mechanism(members, members, y[:-1])


def test_frozen_weight_layer_applies_fixed_weights() -> None:
    members, y = _members(n_members=1, n_rows=600)
    proba = members[0]
    layer = frozen_weight_layer(np.array([1.0, 1.0, 1.0]))
    assert layer(y, proba, y, proba) == pytest.approx(argmax_layer(y, proba, y, proba))
    biased = frozen_weight_layer(np.array([1.0, 1e-9, 1e-9]))
    assert biased(y, proba, y, proba) < 0.4  # everything forced into class 0


def test_selection_null_arms_returns_disjoint_halves_ordered_by_the_m1_read() -> None:
    members, y = _members(n_members=6, n_rows=800)

    challenger, stock = selection_null_arms(members, y, seed=1, layer=argmax_layer)

    assert challenger.shape == stock.shape == (3, 800, 3)
    pooled = np.concatenate([challenger, stock])
    # disjoint: no member appears in both halves
    assert len({tuple(m.ravel()[:8].tolist()) for m in pooled}) == 6
    at_one = [
        np.mean([argmax_layer(y, arm[i], y, arm[i]) for i in range(3)])
        for arm in (challenger, stock)
    ]
    assert at_one[0] >= at_one[1]


def test_mean_shift_control_interpolates_toward_the_truth() -> None:
    members, y = _members(n_members=2, n_rows=400)
    shifted = mean_shift_control(members, y, lam=1.0)
    assert np.allclose(shifted[0], np.eye(3)[y])
    assert np.allclose(mean_shift_control(members, y, lam=0.0), members)


def test_variance_reduction_control_preserves_the_mean_and_cuts_the_spread() -> None:
    pool, _ = _members(n_members=8, n_rows=800)
    arm, target = pool[:4], pool[4:]
    shrunk = variance_reduction_control(arm, target, lam=0.5)
    assert shrunk.std(axis=0).mean() < arm.std(axis=0).mean()
    assert np.allclose(shrunk.sum(axis=2), 1.0, atol=1e-9)


def test_calibrate_lambda_bisects_a_monotone_control() -> None:
    assert calibrate_lambda(lambda lam: lam**2, 0.25) == pytest.approx(0.5, abs=1e-4)
