"""Equivalence tests: package optimizers vs the pre-refactor competition code.

The competition scripts can't be re-run here (data is gitignored), so these
tests pin the refactor semantically: the package functions must reproduce the
exact formulas the s6e5/s6e6 scripts used before the consolidation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data_science_stuff.kaggle.blending import blend, optimize_blend_weights
from data_science_stuff.kaggle.decision import cascade_combine


def _multiclass_fixture() -> "tuple[list[NDArray[np.float64]], NDArray[np.int64]]":
    rng = np.random.default_rng(7)
    n = 300
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3)
    oofs = []
    for seed in (1, 2, 3):
        local = np.random.default_rng(seed)
        proba = np.full((n, 3), 0.15)
        proba[np.arange(n), y] = 0.7
        proba += local.normal(0, 0.2, proba.shape)
        proba = np.clip(proba, 1e-3, None)
        oofs.append(proba / proba.sum(axis=1, keepdims=True))
    del rng
    return oofs, y


def _old_s6e6_optimizer(
    oof_list: "list[NDArray[np.float64]]", y: "NDArray[np.int64]"
) -> "NDArray[np.float64]":
    """Verbatim pre-refactor s6e6 ensemble.py blend-weight search."""

    def old_blend(weights: "NDArray[np.float64]") -> "NDArray[np.float64]":
        w = np.exp(weights) / np.exp(weights).sum()
        return np.asarray(
            sum(w[i] * a for i, a in enumerate(oof_list)), dtype=np.float64
        )

    def objective(weights: "NDArray[np.float64]") -> float:
        return -float(balanced_accuracy_score(y, np.argmax(old_blend(weights), axis=1)))

    result = minimize(
        objective,
        np.ones(len(oof_list)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    return np.asarray(np.exp(result.x) / np.exp(result.x).sum(), dtype=np.float64)


def test_blend_optimizer_matches_old_s6e6_softmax_search() -> None:
    oofs, y = _multiclass_fixture()

    def score(y_true: "NDArray[np.int64]", blended: "NDArray[np.float64]") -> float:
        return float(balanced_accuracy_score(y_true, np.argmax(blended, axis=1)))

    old_w = _old_s6e6_optimizer(oofs, y)
    new_w = optimize_blend_weights(oofs, y, score, normalize="softmax")

    # Same objective landscape and same start → the achieved score must match;
    # the weight vectors agree up to optimizer plateau tolerance.
    old_score = score(y, blend(oofs, old_w))
    new_score = score(y, blend(oofs, new_w))
    assert new_score >= old_score - 1e-9
    np.testing.assert_allclose(new_w, old_w, atol=0.05)


def _old_s6e5_optimizer(
    oofs: "list[NDArray[np.float64]]", y: "NDArray[np.int64]"
) -> "NDArray[np.float64]":
    """Verbatim pre-refactor s6e5 ensemble.py optimize_weights (clip + AUC)."""

    def neg_ensemble_auc(weights: "NDArray[np.float64]") -> float:
        w = np.clip(weights, 0, None)
        w = w / w.sum()
        blended = sum(wi * oof for wi, oof in zip(w, oofs))
        return -float(roc_auc_score(y, blended))

    result = minimize(
        neg_ensemble_auc,
        np.ones(len(oofs)) / len(oofs),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    w = np.clip(result.x, 0, None)
    return np.asarray(w / w.sum(), dtype=np.float64)


def test_blend_optimizer_matches_old_s6e5_clip_search() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    oofs = [np.clip(y + rng.normal(0, noise, n), 0, 1) for noise in (0.3, 0.5, 0.9)]

    def score(y_true: "NDArray[np.int64]", blended: "NDArray[np.float64]") -> float:
        return float(roc_auc_score(y_true, blended))

    old_w = _old_s6e5_optimizer(oofs, y)
    new_w = optimize_blend_weights(
        oofs, y, score, normalize="clip", x0=np.ones(len(oofs)) / len(oofs)
    )

    old_score = score(y, blend(oofs, old_w))
    new_score = score(y, blend(oofs, new_w))
    assert new_score >= old_score - 1e-9
    np.testing.assert_allclose(new_w, old_w, atol=0.05)


def test_cascade_combine_matches_old_s6e6_combine() -> None:
    """Verbatim pre-refactor train_chain_cascade.combine (GAL/QSO/STAR = 0/1/2)."""
    gal, qso, star = 0, 1, 2
    rng = np.random.default_rng(3)
    p_qso = rng.random(200)
    p_star_cond = rng.random(200)

    rest = 1.0 - p_qso
    old = np.empty((len(p_qso), 3), dtype=np.float32)
    old[:, gal] = rest * (1.0 - p_star_cond)
    old[:, qso] = p_qso
    old[:, star] = rest * p_star_cond

    new = cascade_combine([p_qso, p_star_cond], [qso, star, gal])
    np.testing.assert_allclose(new, old, rtol=1e-6)
