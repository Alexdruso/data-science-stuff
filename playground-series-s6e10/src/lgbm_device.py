"""Shared LightGBM device detection for PS S6E10 — package re-export.

The probe lives in `data_science_stuff.kaggle.device.get_lgbm_device`
(probe a tiny `device_type="cuda"` fit once, cached, fall back to
`("cpu", -1)`; never the OpenCL `device="gpu"` backend). Imported by both
the tune and train scripts so the search and the final retrain always
agree on device.
"""

from data_science_stuff.kaggle.device import get_lgbm_device

__all__ = ["get_lgbm_device"]
