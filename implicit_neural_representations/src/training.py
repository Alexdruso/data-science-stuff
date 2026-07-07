"""Training loop and continuous rendering for coordinate networks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from .image import FloatImage


@dataclass
class FitResult:
    """Loss/PSNR trajectory of one full-batch fit."""

    losses: list[float] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)

    @property
    def final_psnr(self) -> float:
        return self.psnrs[-1]


def make_coordinate_grid(
    height: int, width: int, device: torch.device | None = None
) -> Tensor:
    """Pixel-centre coordinates on [-1, 1]^2, shaped (height * width, 2).

    Column 0 is the vertical (row) coordinate, column 1 the horizontal (column)
    coordinate. Using pixel centres (half-pixel offset) makes grids of different
    resolutions sample the same continuous domain consistently, which is what
    lets one trained network render at any resolution.
    """
    ys = (torch.arange(height, device=device) + 0.5) / height * 2.0 - 1.0
    xs = (torch.arange(width, device=device) + 0.5) / width * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)


def fit(
    model: nn.Module,
    coords: Tensor,
    targets: Tensor,
    steps: int = 1500,
    lr: float = 1e-3,
    seed: int = 0,
) -> FitResult:
    """Fit a coordinate network to (coords, targets) with full-batch Adam + MSE.

    ``targets`` are RGB values in [0, 1] shaped (N, 3), matching ``coords``
    shaped (N, 2). The whole image is one batch: these examples are small
    enough that full-batch gradient descent is both simplest and fastest.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = FitResult()
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = model(coords)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        mse = float(loss.detach())
        result.losses.append(mse)
        result.psnrs.append(-10.0 * float(np.log10(max(mse, 1e-12))))
    del optimizer
    if torch.cuda.is_available():  # GPU memory rule: free per-fit allocations
        torch.cuda.empty_cache()
    return result


def render(model: nn.Module, height: int, width: int) -> FloatImage:
    """Query the continuous image f(x, y) on an arbitrary grid.

    This is the payoff of an implicit representation: the same trained weights
    can be sampled at any resolution (including non-integer scale factors),
    because the network represents the image as a continuous function rather
    than a fixed grid of pixels.
    """
    device = next(model.parameters()).device
    coords = make_coordinate_grid(height, width, device=device)
    with torch.no_grad():
        prediction = model(coords).clamp(0.0, 1.0)
    rendered: FloatImage = (
        prediction.reshape(height, width, 3).cpu().numpy().astype(np.float32)
    )
    del coords, prediction
    if torch.cuda.is_available():  # GPU memory rule: free per-render allocations
        torch.cuda.empty_cache()
    return rendered
