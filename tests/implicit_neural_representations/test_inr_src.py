"""Smoke tests for the implicit neural representations example modules."""

import math

import numpy as np
import pytest
import torch

from implicit_neural_representations.src import (
    FourierFeatureMLP,
    ReluMLP,
    Siren,
    downsample,
    fit,
    make_coordinate_grid,
    psnr,
    render,
)
from implicit_neural_representations.src.models import SineLayer


@pytest.mark.parametrize("model_cls", [ReluMLP, FourierFeatureMLP, Siren])
def test_models_map_coords_to_rgb(model_cls):
    model = model_cls(hidden_features=16, hidden_layers=2)
    out = model(torch.rand(5, 2) * 2 - 1)
    assert out.shape == (5, 3)


def test_siren_hidden_init_bounds_follow_paper():
    model = Siren(hidden_features=32, hidden_layers=2, omega_0=30.0)
    hidden = model.net[1]
    assert isinstance(hidden, SineLayer)
    bound = math.sqrt(6.0 / 32) / 30.0
    assert hidden.linear.weight.abs().max().item() <= bound


def test_coordinate_grid_uses_pixel_centers_in_unit_square():
    grid = make_coordinate_grid(4, 8)
    assert grid.shape == (32, 2)
    assert grid.min().item() > -1.0
    assert grid.max().item() < 1.0
    # Rows (column 0) sweep slower than columns (column 1) in a raster scan.
    assert torch.allclose(grid[:8, 0], grid[0, 0].expand(8))


def test_fit_reduces_loss_and_render_matches_shape():
    torch.manual_seed(0)
    target = torch.rand(8, 8, 3)
    coords = make_coordinate_grid(8, 8)
    model = Siren(hidden_features=16, hidden_layers=2)
    result = fit(model, coords, target.reshape(-1, 3), steps=50, lr=1e-3)
    assert result.losses[-1] < result.losses[0]
    assert len(result.psnrs) == 50

    image = render(model, 16, 16)
    assert image.shape == (16, 16, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_downsample_area_average_and_psnr():
    img = (np.arange(4 * 4 * 3) / (4 * 4 * 3)).reshape(4, 4, 3).astype(np.float32)
    small = downsample(img, 2)
    assert small.shape == (2, 2, 3)
    assert np.allclose(small[0, 0], img[:2, :2].mean(axis=(0, 1)))
    assert psnr(img, img) == math.inf
    assert psnr(img, np.zeros_like(img)) > 0.0
