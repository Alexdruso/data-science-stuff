"""Shared LightGBM device detection for PS S6E7.

Probes once for a working CUDA build and falls back to CPU, returning the matching
`(device_type, n_jobs)` pair: GPU wants `n_jobs=1` (CPU threads add overhead), CPU
wants `n_jobs=-1` (use all cores). Imported by every LGBM script so all runs agree
on device.

2026-07-02: lightgbm 4.6.0 rebuilt from source with `-DUSE_CUDA=1` against the
user-space toolkit at ~/.local/cuda-12.4 — the probe now returns ("cuda", 1).
NOTE: on this dataset (13 features, 690k rows) CUDA is a WASH vs 12-core CPU
(~123s vs ~90-150s/fold); its value is freeing the CPU to train another model
concurrently (e.g. HGBC), not raw speed.
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
        LGBMClassifier(n_estimators=1, device_type="cuda", verbose=-1, n_jobs=1).fit(
            x, y
        )
        _CACHED = ("cuda", 1)
    except Exception:  # noqa: BLE001 — any CUDA-build failure means fall back to CPU
        _CACHED = ("cpu", -1)
    return _CACHED
