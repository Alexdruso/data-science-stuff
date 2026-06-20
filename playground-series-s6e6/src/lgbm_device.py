"""Shared LightGBM device detection for PS S6E6.

The pip `lightgbm` wheel in this environment is CPU-only (not built with
`-DUSE_CUDA=1`), so `device_type="cuda"` raises at fit time. This helper probes
for a working CUDA build once and falls back to CPU, returning the matching
`(device_type, n_jobs)` pair: GPU wants `n_jobs=1` (CPU threads add overhead),
CPU wants `n_jobs=-1` (use all cores). Imported by both `tune_lgbm.py` and
`baseline.py` so the tuning search and the final retrain always agree on device.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

_CACHED: tuple[str, int] | None = None


def get_lgbm_device() -> tuple[str, int]:
    """Return (device_type, n_jobs): ("cuda", 1) if a CUDA build works, else ("cpu", -1)."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    try:
        LGBMClassifier(
            n_estimators=1, device_type="cuda", verbose=-1, n_jobs=1
        ).fit(x, y)
        _CACHED = ("cuda", 1)
    except Exception:  # noqa: BLE001 — any CUDA-build failure means fall back to CPU
        _CACHED = ("cpu", -1)
    return _CACHED
