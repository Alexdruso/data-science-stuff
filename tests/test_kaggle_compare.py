"""Tests for data_science_stuff.kaggle.compare."""

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from data_science_stuff.kaggle.compare import (
    honest_blend_delta,
    honest_blend_score,
    paired_seed_deltas,
)


def test_paired_seed_deltas_pairs_on_shared_seeds_only() -> None:
    scores = {
        "base": {1: 0.900, 2: 0.910, 3: 0.920},
        "good": {1: 0.902, 2: 0.913, 3: 0.921},
        "partial": {2: 0.905, 9: 0.999},
        "disjoint": {7: 0.95},
    }
    out = paired_seed_deltas(scores, "base")
    rows = out.to_dict("index")
    assert rows["good"]["n_seeds"] == 3
    assert rows["good"]["delta"] == pytest.approx(0.002)
    assert rows["good"]["t"] > 0
    assert rows["partial"]["n_seeds"] == 1
    assert rows["partial"]["delta"] == pytest.approx(-0.005)
    assert math.isnan(rows["partial"]["sd_delta"])
    assert math.isnan(rows["disjoint"]["delta"])
    assert math.isnan(rows["base"]["delta"])
    assert out.index[0] == "good"  # sorted best first


def test_paired_seed_deltas_unknown_baseline() -> None:
    with pytest.raises(KeyError):
        paired_seed_deltas({"a": {1: 0.5}}, "b")


_Arr = NDArray[Any]


def _toy(seed: int = 0) -> "tuple[_Arr, _Arr, _Arr, _Arr]":
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 1.0, n)
    base = signal + rng.normal(0, 1.0, n)
    informative = signal + rng.normal(0, 1.0, n)  # independent noise: helps
    noise = rng.normal(0, 1.0, n)  # no information
    return y, base, informative, noise


def test_honest_blend_rewards_information_not_noise() -> None:
    y, base, informative, noise = _toy()
    solo = roc_auc_score(y, base)
    good = honest_blend_score(y, base, informative, seeds=(0, 1))
    bad = honest_blend_score(y, base, noise, seeds=(0, 1))
    assert good.score > solo + 0.01
    assert all(0.3 <= w <= 0.7 for w in good.weights)
    assert bad.score <= solo + 0.002
    assert len(good.half_scores) == 4


def test_honest_blend_delta_sign() -> None:
    y, base, informative, noise = _toy(1)
    assert honest_blend_delta(y, base, informative, noise) > 0.01
    assert honest_blend_delta(y, base, noise, informative) < -0.01
