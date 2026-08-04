"""Tests for data_science_stuff.kaggle.shift (covariate + mask-mechanism shift)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from data_science_stuff.kaggle.shift import (
    adversarial_validation,
    importance_weights,
    mask_frame,
    mask_mechanism_report,
    unseen_mask_patterns,
    weighted_balanced_accuracy,
)


def _tiny_tree() -> Any:
    """A fast stand-in for the LightGBM default (keeps the suite quick)."""
    return DecisionTreeClassifier(max_depth=4, random_state=0)


def _tiny_lgbm() -> Any:
    """The real default family, sized down — the only one that eats `category`."""
    return LGBMClassifier(n_estimators=8, num_leaves=7, verbose=-1, random_state=0)


def test_adversarial_validation_is_blind_when_the_splits_match() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    test = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})

    result = adversarial_validation(train, test, estimator_factory=_tiny_tree)

    assert result.auc == pytest.approx(0.5, abs=0.12)
    assert result.train_p_test.shape == (300,)
    assert result.test_p_test.shape == (300,)


def test_adversarial_validation_detects_a_shifted_column() -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    test = pd.DataFrame({"a": rng.normal(size=300) + 4.0, "b": rng.normal(size=300)})

    result = adversarial_validation(train, test, estimator_factory=_tiny_tree)

    assert result.auc > 0.9
    assert result.test_p_test.mean() > result.train_p_test.mean()


def test_adversarial_validation_uses_shared_columns_and_casts_categoricals() -> None:
    rng = np.random.default_rng(2)
    train = pd.DataFrame(
        {"a": rng.normal(size=120), "g": ["x", "y"] * 60, "only_train": 1.0}
    )
    test = pd.DataFrame({"a": rng.normal(size=120), "g": ["x", "y"] * 60})

    result = adversarial_validation(
        train, test, cat_cols=["g"], n_splits=3, estimator_factory=_tiny_lgbm
    )

    assert 0.0 <= result.auc <= 1.0


def test_adversarial_validation_rejects_disjoint_frames() -> None:
    with pytest.raises(ValueError, match="share no columns"):
        adversarial_validation(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))


def test_mask_frame_marks_missing_cells() -> None:
    frame = pd.DataFrame({"a": [1.0, None], "b": [None, None]})
    assert mask_frame(frame).to_numpy().tolist() == [[False, True], [True, True]]


def test_mask_mechanism_report_flags_a_mechanism_shift_with_matched_rates() -> None:
    """The s6e7 case: identical marginal null rates, opposite mask mechanisms."""
    rng = np.random.default_rng(3)
    n = 400
    trigger = rng.integers(0, 2, n).astype(bool)
    # train: `value` is missing exactly when `trigger` is set (a rule, AUC -> 1)
    train = pd.DataFrame(
        {
            "trigger": np.where(trigger, np.nan, 1.0),
            "value": np.where(trigger, np.nan, 1.0),
        }
    )
    # test: same marginal rate, but missing uniformly at random (AUC -> 0.5)
    uniform = rng.permutation(trigger)
    test = pd.DataFrame(
        {
            "trigger": np.where(trigger, np.nan, 1.0),
            "value": np.where(uniform, np.nan, 1.0),
        }
    )

    report = mask_mechanism_report(train, test, estimator_factory=_tiny_tree)

    row = report[report["column"] == "value"].iloc[0]
    assert row["train_rate"] == pytest.approx(row["test_rate"])  # rates match exactly
    assert row["train_auc"] > 0.95  # ...and the mechanisms do not
    assert row["test_auc"] == pytest.approx(0.5, abs=0.15)
    assert report["auc_gap"].is_monotonic_decreasing  # largest gap first
    assert bool((report["auc_gap"] > 0.3).all())  # both masks changed mechanism


def test_mask_mechanism_report_returns_nan_when_a_split_has_no_missingness() -> None:
    train = pd.DataFrame({"a": [1.0, None, 1.0, None], "b": [1.0] * 4})
    test = pd.DataFrame({"a": [1.0] * 4, "b": [1.0] * 4})

    report = mask_mechanism_report(
        train, test, n_splits=2, estimator_factory=_tiny_tree
    )

    assert list(report["column"]) == ["a"]
    assert np.isnan(report.iloc[0]["test_auc"])


def test_mask_mechanism_report_is_empty_without_missing_values() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert mask_mechanism_report(frame, frame).empty


def test_unseen_mask_patterns_finds_train_impossible_rows() -> None:
    train = pd.DataFrame({"a": [1.0, None, 1.0], "b": [1.0, 1.0, None]})
    test = pd.DataFrame({"a": [None, 1.0], "b": [None, 1.0]})

    unseen = unseen_mask_patterns(train, test)

    assert unseen.tolist() == [True, False]  # co-missing never occurs in train


def test_test_like_weights_are_monotone_and_clipped() -> None:
    weights = importance_weights(np.array([0.1, 0.5, 0.9, 0.999]), clip=5.0)
    assert weights[0] < weights[1] < weights[2]
    assert weights.max() == 5.0


def test_weighted_balanced_accuracy_matches_sklearn_at_uniform_weights() -> None:
    rng = np.random.default_rng(4)
    y = np.asarray(rng.integers(0, 3, 200), dtype=np.int64)
    pred = np.where(rng.random(200) < 0.7, y, (y + 1) % 3).astype(np.int64)

    assert weighted_balanced_accuracy(y, pred, np.ones(200)) == pytest.approx(
        float(balanced_accuracy_score(y, pred))
    )


def test_weighted_balanced_accuracy_follows_the_weights() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 1, 1, 1], dtype=np.int64)
    # Weighting the correctly classified class-0 row to zero drops its recall.
    assert weighted_balanced_accuracy(
        y, pred, np.array([0.0, 1.0, 1.0, 1.0])
    ) == pytest.approx(0.5)
