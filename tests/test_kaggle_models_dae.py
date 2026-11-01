"""Tests for data_science_stuff.kaggle.models.dae."""

import numpy as np
import torch

from data_science_stuff.kaggle.models.dae import SwapNoiseDAE, fit_dae_features


def test_dae_forward_shapes() -> None:
    model = SwapNoiseDAE(5, 8)
    recon, hidden = model(torch.zeros(3, 5))
    assert recon.shape == (3, 5)
    assert hidden.shape == (3, 24)


def test_fit_dae_features_shape_dtype_and_determinism() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6)).astype(np.float32)
    kwargs = {"hidden": 8, "epochs": 2, "batch_size": 64, "device": torch.device("cpu")}
    a = fit_dae_features(x, seed=1, **kwargs)  # type: ignore[arg-type]
    b = fit_dae_features(x, seed=1, **kwargs)  # type: ignore[arg-type]
    assert a.shape == (300, 24)
    assert a.dtype == np.float32
    assert (a >= 0).all()  # ReLU activations
    np.testing.assert_allclose(a, b)
