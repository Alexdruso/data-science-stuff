"""LightGBM device auto-detection."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from lightgbm import LGBMClassifier


@lru_cache(maxsize=1)
def get_lgbm_device() -> tuple[str, int]:
    """Return ``(device_type, n_jobs)``: ``("cuda", 1)`` if usable, else ``("cpu", -1)``.

    pip ``lightgbm`` wheels are usually CPU-only (not built with
    ``-DUSE_CUDA=1``), so ``device_type="cuda"`` raises at fit time rather
    than at import. This probes a tiny fit once (cached) and falls back to
    CPU, returning the matching pair: GPU wants ``n_jobs=1`` (CPU threads add
    overhead), CPU wants ``n_jobs=-1`` (use all cores). Share the result
    between train and tune scripts so the search and the final retrain always
    agree on device.

    Never use ``device="gpu"`` (the OpenCL backend): it is slow or silently
    falls back on NVIDIA. Only the ``"cuda"`` device type is worth probing.
    """
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    try:
        LGBMClassifier(n_estimators=1, device_type="cuda", verbose=-1, n_jobs=1).fit(
            x, y
        )
    except Exception:  # noqa: BLE001 — any CUDA-build failure means fall back to CPU
        return ("cpu", -1)
    return ("cuda", 1)
