"""cdeotte's from-scratch RealMLP for S6E6 — package re-export.

The full implementation (PBLD embeddings, NTP linears, internal ensemble,
EMA, `loss_prior_power` logit adjustment) was extracted verbatim to
`data_science_stuff.kaggle.models.realmlp`. This shim keeps the existing
`from realmlp_deotte import CONFIG, RealMLP_TD_Classifier, ...` call sites
in the train_realmlp_*.py scripts working unchanged. `CONFIG` is the
package's `REALMLP_TD_CONFIG` (defaults preserved, including
`loss_prior_power=1.075`).
"""

from data_science_stuff.kaggle.models.losses import smooth_ce_loss
from data_science_stuff.kaggle.models.realmlp import (
    DEVICE,
    REALMLP_TD_CONFIG as CONFIG,
    CategoricalFeatureLayer,
    NTPLinear,
    NumericalPreprocessor,
    PBLDEmbedding,
    RealMLP,
    RealMLP_TD_Classifier,
    ScalingLayer,
    apply_schedule,
    get_parameter_groups,
)

__all__ = [
    "CONFIG",
    "DEVICE",
    "CategoricalFeatureLayer",
    "NTPLinear",
    "NumericalPreprocessor",
    "PBLDEmbedding",
    "RealMLP",
    "RealMLP_TD_Classifier",
    "ScalingLayer",
    "apply_schedule",
    "get_parameter_groups",
    "smooth_ce_loss",
]
