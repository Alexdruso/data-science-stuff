"""Swap-noise denoising autoencoder: label-free representation features.

Extracted from s6e6 ``train_dae.py`` and s6e10 ``dae_leg.py`` (two separate
implementations of the same idea). The DAE is trained on train + test rows
together without labels, so it may see every row; its concatenated hidden
activations become the input of a supervised head (an MLP or a GBDT) that is
then trained fold by fold as usual.

Inputs should be roughly standardised: one-hot blocks for categoricals and
:func:`data_science_stuff.kaggle.encoding.rank_gauss` for numerics.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


class SwapNoiseDAE(nn.Module):
    """Three equal-width ReLU layers and a linear reconstruction head."""

    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_features, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, concatenated hidden activations)``."""
        a1 = torch.relu(self.l1(x))
        a2 = torch.relu(self.l2(a1))
        a3 = torch.relu(self.l3(a2))
        return self.out(a3), torch.cat([a1, a2, a3], dim=1)


def fit_dae_features(
    x: NDArray[np.float32],
    *,
    hidden: int = 512,
    epochs: int = 30,
    noise: float = 0.15,
    lr: float = 3e-4,
    batch_size: int = 1024,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> NDArray[np.float32]:
    """Train a swap-noise DAE on ``x`` and return its hidden activations.

    Swap noise: with probability ``noise`` each cell is replaced by the same
    column of a random other row, so the corruption stays on the column's own
    marginal (Gaussian noise would not, for one-hot or rank-gauss inputs).
    The loss is MSE between the reconstruction and the clean row.

    Args:
        x: (n_rows, n_features) float32 inputs — train and test rows stacked.
        hidden: Width of each of the three hidden layers.
        epochs: Training epochs (fixed: there is no validation split).
        noise: Per-cell swap probability.
        lr: Adam learning rate.
        batch_size: Minibatch size.
        seed: Torch and numpy seed.
        device: Defaults to CUDA when available.
        verbose: Print the reconstruction loss each epoch.

    Returns:
        (n_rows, 3 * hidden) float32 activations, in the row order of ``x``.
        Slice train/test rows back out by position.

    GPU tensors and the model are deleted and the CUDA cache emptied before
    returning (GPU memory rule).
    """
    dev = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    n, d = x_t.shape
    model = SwapNoiseDAE(d, hidden).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = x_t[perm[i : i + batch_size]]
            mask = torch.rand_like(batch) < noise
            donor = x_t[torch.randint(0, n, (len(batch),), device=dev)]
            recon, _ = model(torch.where(mask, donor, batch))
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if verbose:
            print(f"  dae epoch {epoch + 1}/{epochs}: loss {total / n:.5f}")

    model.eval()
    feats = np.zeros((n, 3 * hidden), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, 8192):
            feats[i : i + 8192] = model(x_t[i : i + 8192])[1].cpu().numpy()

    del model, optimizer, x_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return feats
