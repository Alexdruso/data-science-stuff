"""Tests for data_science_stuff.kaggle.models.mlp (fast CPU smoke)."""

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_class_weight

from data_science_stuff.kaggle.models.losses import logit_adjustment
from data_science_stuff.kaggle.models.mlp import (
    MLP,
    build_layer_sizes,
    class_weights,
    fit_mlp_fold,
    ohe_feature_matrix,
)

_PARAMS: "dict[str, object]" = {
    "n_layers": 2,
    "first_size": 16,
    "shrink": 0.5,
    "dropout": 0.1,
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "activation": "relu",
}


def test_build_layer_sizes_shrinks_with_floor() -> None:
    assert build_layer_sizes(3, 512) == [512, 256, 128]
    assert build_layer_sizes(4, 100, 0.5) == [100, 50, 32, 32]  # floored at 32
    assert build_layer_sizes(1, 64) == [64]


def test_mlp_forward_shapes_and_activations() -> None:
    torch.manual_seed(0)
    for activation in ("relu", "gelu"):
        model = MLP(8, [16, 8], 0.1, n_classes=3, activation=activation)
        out = model(torch.randn(4, 8))
        assert out.shape == (4, 3)


def test_class_weights_matches_sklearn_balanced() -> None:
    y = np.array([0] * 70 + [1] * 20 + [2] * 10, dtype=np.int64)
    ours = class_weights(y, 3)
    reference = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1, 2]), y=y
    )
    np.testing.assert_allclose(ours, reference.astype(np.float32), rtol=1e-6)


def test_ohe_feature_matrix_shapes_and_indices() -> None:
    train = pd.DataFrame(
        {"num1": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"], "num2": [0.1, 0.2, 0.3]}
    )
    test = pd.DataFrame({"num1": [4.0], "cat": ["c"], "num2": [0.4]})

    x, x_test, num_idx, cat_idx = ohe_feature_matrix(
        train, test, ["cat"], ["num1", "cat", "num2"]
    )

    # 2 numeric + 3 OHE columns (categories a, b, c seen across train+test).
    assert x.shape == (3, 5)
    assert x_test.shape == (1, 5)
    assert num_idx == [0, 1]
    assert cat_idx == [2, 3, 4]
    assert x.dtype == np.float32
    # The test-only category "c" still gets a column, zero in train.
    assert x[:, cat_idx].sum(axis=0).tolist() == [2.0, 1.0, 0.0]


def _fold_data() -> (
    "tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32],"
    " NDArray[np.int64], NDArray[np.float32]]"
):
    rng = np.random.default_rng(0)
    n = 200
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3 + 1)[:n]
    x = rng.normal(0, 1, (n, 6)).astype(np.float32)
    x[np.arange(n), y * 2] += 3.0  # separable signal in columns 0/2/4
    split = 150
    return x[:split], y[:split], x[split:], y[split:], x[:20]


def test_fit_mlp_fold_smoke_with_class_weights() -> None:
    torch.manual_seed(0)
    x_tr, y_tr, x_va, y_va, x_test = _fold_data()

    val_pred, test_pred = fit_mlp_fold(
        x_tr,
        y_tr,
        x_va,
        y_va,
        x_test,
        num_idx=[0, 1, 2, 3],
        cat_idx=[4, 5],
        params=_PARAMS,
        n_classes=3,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=64,
        patience=2,
    )

    assert val_pred.shape == (len(x_va), 3)
    assert test_pred.shape == (20, 3)
    np.testing.assert_allclose(val_pred.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(test_pred.sum(axis=1), 1.0, atol=1e-5)


def test_fit_mlp_fold_smoke_with_logit_adjustment() -> None:
    torch.manual_seed(0)
    x_tr, y_tr, x_va, y_va, x_test = _fold_data()
    adj = logit_adjustment(np.bincount(y_tr, minlength=3), tau=1.075)

    val_pred, test_pred = fit_mlp_fold(
        x_tr,
        y_tr,
        x_va,
        y_va,
        x_test,
        num_idx=list(range(6)),
        cat_idx=[],
        params=_PARAMS,
        n_classes=3,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=64,
        patience=2,
        use_class_weights=False,
        logit_adjust=adj,
    )

    assert val_pred.shape == (len(x_va), 3)
    np.testing.assert_allclose(val_pred.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(test_pred).all()
