"""GBDT device selection: LightGBM CUDA probe, XGBoost and CatBoost GPU params."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

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


@lru_cache(maxsize=1)
def get_xgb_device() -> str:
    """Return ``"cuda"`` when XGBoost can train on a GPU here, else ``"cpu"``.

    The ``XGB_DEVICE`` environment variable overrides the probe. Otherwise
    this checks that the XGBoost build has CUDA and that torch sees a device
    (XGBoost itself only warns and silently falls back when it has neither).
    """
    override = os.environ.get("XGB_DEVICE")
    if override:
        return override
    import xgboost  # noqa: PLC0415 — keep import cost off the LightGBM path

    if not xgboost.build_info().get("USE_CUDA", False):
        return "cpu"
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover — torch is a declared dependency
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def xgb_device_params() -> dict[str, Any]:
    """``{"tree_method": "hist", "device": ...}`` to merge into XGBoost params.

    Replaces the ``"device": "cuda"`` hard-coded in 18 competition scripts,
    which crashed or silently ran on CPU on machines without a GPU.
    """
    return {"tree_method": "hist", "device": get_xgb_device()}


def catboost_gpu_params(
    task_type: str | None = None, gpu_ram_part: float | None = None
) -> dict[str, Any]:
    """CatBoost device params with the fixes measured on a 6 GB card.

    ``task_type`` defaults to the ``CB_TASK_TYPE`` env var, else ``"GPU"``.
    There is deliberately no live probe: a tiny test fit fails transiently
    while the GPU still holds memory from a just-killed process, which then
    sends a whole run to CPU.

    On GPU three things are set (all measured on s6e6-s6e8):

    - ``border_count=254``: the GPU default is 128 (the CPU default is 254).
      Halving split resolution cost s6e8 0.0027 OOF AUC with no warning, and
      254 was also faster. Set on CPU too, so both devices fit the same model.
    - ``max_ctr_complexity=1``: the default (2) builds categorical-combination
      CTR tables that overflow 6 GB at ~600k rows, and CatBoost then
      *silently* falls back to CPU (~20x slower).
    - ``gpu_ram_part`` (``CB_GPU_RAM_PART`` env var, default 0.85): lower it
      (~0.35) when another job shares the GPU, or that job gets OOM-killed.

    The other half of the fix is in the caller: **do not pass an
    ``eval_set`` on GPU** (a second CTR set overflows memory again). Train a
    fixed number of ``iterations`` and predict the validation split after.
    """
    task = task_type or os.environ.get("CB_TASK_TYPE", "GPU")
    params: dict[str, Any] = {"task_type": task, "border_count": 254}
    if task == "GPU":
        ram = (
            gpu_ram_part
            if gpu_ram_part is not None
            else float(os.environ.get("CB_GPU_RAM_PART", "0.85"))
        )
        params.update({"max_ctr_complexity": 1, "gpu_ram_part": ram})
    return params
