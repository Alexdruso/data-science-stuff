"""Tests for data_science_stuff.kaggle.device."""

from collections.abc import Iterator
from typing import Any

import pytest

from data_science_stuff.kaggle import device


@pytest.fixture(autouse=True)
def _fresh_probe() -> Iterator[None]:
    """Reset the lru_cache around every test so probes don't leak between them."""
    device.get_lgbm_device.cache_clear()
    yield
    device.get_lgbm_device.cache_clear()


_CUDA_BUILD_ERROR = "CUDA Tree Learner was not enabled in this build"


class _CudaOk:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, _x: Any, _y: Any) -> "_CudaOk":
        return self


class _CudaBroken:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, _x: Any, _y: Any) -> None:
        raise RuntimeError(_CUDA_BUILD_ERROR)


def test_cpu_fallback_when_cuda_fit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "LGBMClassifier", _CudaBroken)
    assert device.get_lgbm_device() == ("cpu", -1)


def test_cuda_when_probe_fit_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "LGBMClassifier", _CudaOk)
    assert device.get_lgbm_device() == ("cuda", 1)


def test_probe_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "LGBMClassifier", _CudaOk)
    assert device.get_lgbm_device() == ("cuda", 1)
    # Swapping the classifier afterwards must not change the cached answer.
    monkeypatch.setattr(device, "LGBMClassifier", _CudaBroken)
    assert device.get_lgbm_device() == ("cuda", 1)
