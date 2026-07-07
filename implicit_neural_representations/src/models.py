"""Coordinate networks: ReLU MLP, Fourier-feature MLP (Tancik et al., 2020) and SIREN.

All models share one interface: they map a batch of 2-D coordinates in
[-1, 1]^2, shaped (N, 2), to RGB values shaped (N, 3).

References:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation
    Functions" (SIREN), NeurIPS 2020, arXiv:2006.09661.
    Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains", NeurIPS 2020, arXiv:2006.10739.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _mlp(in_features: int, hidden_features: int, hidden_layers: int) -> nn.Sequential:
    """Plain ReLU MLP trunk with a linear RGB head."""
    layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
    layers.append(nn.Linear(hidden_features, 3))
    return nn.Sequential(*layers)


class ReluMLP(nn.Module):
    """Baseline coordinate network: raw (x, y) inputs through a ReLU MLP.

    Spectral bias (Rahaman et al., 2019) makes this model converge to the
    low-frequency content of the image only, so its reconstructions are blurry.
    """

    def __init__(self, hidden_features: int = 128, hidden_layers: int = 3) -> None:
        super().__init__()
        self.net = _mlp(2, hidden_features, hidden_layers)

    def forward(self, coords: Tensor) -> Tensor:
        out: Tensor = self.net(coords)
        return out


class FourierFeatureMLP(nn.Module):
    """ReLU MLP on random Fourier features of the coordinates (Tancik et al., 2020).

    Coordinates are lifted to [sin(2*pi*B x), cos(2*pi*B x)] with a fixed random
    Gaussian matrix B ~ N(0, sigma^2), which lets the downstream ReLU MLP learn
    high-frequency content.
    """

    def __init__(
        self,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        mapping_size: int = 128,
        sigma: float = 10.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        b_matrix = sigma * torch.randn(mapping_size, 2, generator=generator)
        self.b_matrix: Tensor
        self.register_buffer("b_matrix", b_matrix)
        self.net = _mlp(2 * mapping_size, hidden_features, hidden_layers)

    def forward(self, coords: Tensor) -> Tensor:
        projected = 2.0 * math.pi * coords @ self.b_matrix.T
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        out: Tensor = self.net(features)
        return out


class SineLayer(nn.Module):
    """Linear layer followed by sin(omega_0 * x), initialised per the SIREN paper."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        *,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, coords: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(coords))


class Siren(nn.Module):
    """SIREN coordinate network (Sitzmann et al., 2020): an MLP with sine activations.

    The principled initialisation keeps activations distributed so that deep
    stacks of sine layers train stably; omega_0 = 30 follows the paper.
    """

    def __init__(
        self,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(2, hidden_features, omega_0=omega_0, is_first=True)
        ]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
        head = nn.Linear(hidden_features, 3)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega_0
            head.weight.uniform_(-bound, bound)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        out: Tensor = self.net(coords)
        return out
