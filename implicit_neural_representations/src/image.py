"""Image loading and resampling helpers for the implicit functions example."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float32]

_GRAYSCALE_NDIM = 2


def load_image(path: str | Path) -> FloatImage:
    """Load a PNG as a float32 HxWx3 array in [0, 1], dropping any alpha channel."""
    img = np.asarray(plt.imread(path), dtype=np.float32)
    if img.ndim == _GRAYSCALE_NDIM:
        img = np.stack([img] * 3, axis=-1)
    return np.clip(img[..., :3], 0.0, 1.0)


def downsample(img: FloatImage, factor: int) -> FloatImage:
    """Area-average downsampling by an integer factor (crops any remainder rows/cols)."""
    if factor < 1:
        msg = f"factor must be >= 1, got {factor}"
        raise ValueError(msg)
    h, w, c = img.shape
    h, w = h - h % factor, w - w % factor
    blocks = img[:h, :w].reshape(h // factor, factor, w // factor, factor, c)
    averaged: FloatImage = blocks.mean(axis=(1, 3), dtype=np.float32)
    return averaged


def psnr(prediction: FloatImage, target: FloatImage) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]."""
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0.0:
        return math.inf
    return -10.0 * math.log10(mse)
