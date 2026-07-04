"""Tests for data_science_stuff.kaggle.cv."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from data_science_stuff.kaggle.cv import Matrix, run_cv


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float) % 7}
    )


def test_run_cv_fold_bookkeeping_matches_direct_stratified_kfold() -> None:
    n_train, n_test = 100, 40
    x, x_test = _frame(n_train), _frame(n_test)
    y = np.array([0, 1] * 50, dtype=np.int64)
    seen_folds: list[int] = []

    def fit_fold(
        x_tr: Matrix,
        _y_tr: "NDArray[np.int64]",
        x_va: Matrix,
        _y_va: "NDArray[np.int64]",
        x_te: Matrix,
        fold: int,
    ) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
        assert len(x_tr) + len(x_va) == n_train
        return np.full(len(x_va), float(fold)), np.full(len(x_te), float(fold))

    def first_pred(
        _y_va: "NDArray[np.int64]", val_pred: "NDArray[np.float64]"
    ) -> float:
        return float(val_pred[0])

    result = run_cv(
        x, y, x_test, fit_fold, score_fn=first_pred, after_fold=seen_folds.append
    )

    # Every OOF row is written exactly once, by the same folds a direct
    # StratifiedKFold(5, shuffle=True, random_state=42) would produce.
    expected = np.zeros(n_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(n_train), y), 1):
        expected[va_idx] = fold
    np.testing.assert_array_equal(result.oof, expected)

    # Test predictions are the mean over folds: mean(1..5) == 3.
    np.testing.assert_allclose(result.test, np.full(n_test, 3.0))
    assert result.fold_scores == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert seen_folds == [1, 2, 3, 4, 5]


def test_run_cv_multiclass_shapes_with_ndarray_inputs() -> None:
    n_train, n_test = 60, 20
    x = np.arange(n_train * 2, dtype=float).reshape(n_train, 2)
    x_test = np.arange(n_test * 2, dtype=float).reshape(n_test, 2)
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), 20)

    def fit_fold(
        _x_tr: Matrix,
        _y_tr: "NDArray[np.int64]",
        x_va: Matrix,
        _y_va: "NDArray[np.int64]",
        x_te: Matrix,
        _fold: int,
    ) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
        return np.full((len(x_va), 3), 1 / 3), np.full((len(x_te), 3), 1 / 3)

    result = run_cv(x, y, x_test, fit_fold, n_outputs=3)

    assert result.oof.shape == (n_train, 3)
    assert result.test.shape == (n_test, 3)
    np.testing.assert_allclose(result.oof, 1 / 3)
    np.testing.assert_allclose(result.test, 1 / 3)
    assert result.fold_scores == []


def test_run_cv_unstratified_for_continuous_targets() -> None:
    n_train = 50
    x, x_test = _frame(n_train), _frame(10)
    y = np.linspace(0.0, 1.0, n_train)  # StratifiedKFold would reject this

    def fit_fold(
        _x_tr: Matrix,
        y_tr: "NDArray[np.float64]",
        x_va: Matrix,
        _y_va: "NDArray[np.float64]",
        x_te: Matrix,
        _fold: int,
    ) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
        return np.full(len(x_va), float(y_tr.mean())), np.zeros(len(x_te))

    result = run_cv(x, y, x_test, fit_fold, stratified=False)

    assert result.oof.shape == (n_train,)
    assert (result.oof > 0).all()
