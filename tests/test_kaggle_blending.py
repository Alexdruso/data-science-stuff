"""Tests for data_science_stuff.kaggle.blending."""

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data_science_stuff.kaggle.blending import (
    blend,
    diversity_report,
    normalize_weights,
    optimize_blend_weights,
)


def _complementary_pair() -> "tuple[list[NDArray[np.float64]], NDArray[np.int64]]":
    """Two models, each confidently right on one class and wrong on the other."""
    y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    model_a = np.zeros((100, 2))
    model_b = np.zeros((100, 2))
    model_a[:50] = [0.9, 0.1]  # right on class 0
    model_a[50:] = [0.6, 0.4]  # wrong on class 1
    model_b[:50] = [0.4, 0.6]  # wrong on class 0
    model_b[50:] = [0.1, 0.9]  # right on class 1
    return [model_a, model_b], y


def _ba_argmax(y: "NDArray[np.int64]", blended: "NDArray[np.float64]") -> float:
    return float(balanced_accuracy_score(y, np.argmax(blended, axis=1)))


def test_normalize_weights_both_methods_sum_to_one() -> None:
    raw = np.array([2.0, -1.0, 1.0])
    for method in ("clip", "softmax"):
        w = normalize_weights(raw, method=method)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
    # clip can zero a model out entirely; softmax cannot.
    assert normalize_weights(raw, method="clip")[1] == 0.0
    assert normalize_weights(raw, method="softmax")[1] > 0.0


def test_normalize_weights_clip_all_negative_raises() -> None:
    with pytest.raises(ValueError, match="clipped to zero"):
        normalize_weights(np.array([-1.0, -2.0]), method="clip")


def test_optimized_blend_beats_both_singles() -> None:
    oofs, y = _complementary_pair()
    singles = [_ba_argmax(y, o) for o in oofs]
    for method in ("clip", "softmax"):
        w = optimize_blend_weights(oofs, y, _ba_argmax, normalize=method)
        blended_score = _ba_argmax(y, blend(oofs, w))
        assert blended_score > max(singles)
        assert blended_score == pytest.approx(1.0)


def test_optimize_blend_weights_one_dimensional_auc_path() -> None:
    # 1-D binary probabilities scored by AUC — the s6e5 shape.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    informative = np.clip(y + rng.normal(0, 0.3, 200), 0, 1)
    noise = rng.random(200)

    def auc(y_true: "NDArray[np.int64]", blended: "NDArray[np.float64]") -> float:
        return float(roc_auc_score(y_true, blended))

    w = optimize_blend_weights([informative, noise], y, auc, normalize="clip")
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]  # weight concentrates on the informative model


def test_optimize_blend_weights_needs_two_models() -> None:
    with pytest.raises(ValueError, match="at least two"):
        optimize_blend_weights([np.zeros((3, 2))], np.zeros(3), _ba_argmax)


def test_diversity_report_hand_computed() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    # anchor predicts [0, 1, 1, 0] → wrong on rows 1 and 3.
    anchor = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]])
    # other predicts [0, 0, 1, 0] → wrong on row 3 only.
    other = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.6, 0.4]])

    report = diversity_report(
        {"anchor": anchor, "other": other}, y, ["neg", "pos"], anchor="anchor"
    )

    assert list(report.index) == ["anchor", "other"]
    overlap = report[["fixes_anchor", "anchor_fixes"]].to_numpy(dtype=float)
    assert np.isnan(overlap[0]).all()  # anchor row: overlap vs itself is undefined
    # Of the anchor's two errors (rows 1, 3) the other model fixes row 1 only.
    assert report.loc["other", "fixes_anchor"] == pytest.approx(0.5)
    # Of the other's single error (row 3) the anchor fixes none.
    assert report.loc["other", "anchor_fixes"] == pytest.approx(0.0)
    assert report.loc["other", "recall_neg"] == pytest.approx(1.0)
    assert report.loc["other", "recall_pos"] == pytest.approx(0.5)


def test_diversity_report_unknown_anchor_raises() -> None:
    with pytest.raises(ValueError, match="anchor"):
        diversity_report(
            {"m": np.ones((2, 2))}, np.array([0, 1]), ["a", "b"], anchor="missing"
        )
