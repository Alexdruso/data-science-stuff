"""Reusable tabular neural models (PyTorch).

GPU memory rule (mandatory for every consumer): any function that allocates
GPU tensors or models must ``del`` them before returning, and the outer
fold/trial loop must call ``torch.cuda.empty_cache()`` after each iteration —
omitting this OOM-crashes multi-fold/multi-trial runs.
"""

from data_science_stuff.kaggle.models.losses import logit_adjustment, smooth_ce_loss
from data_science_stuff.kaggle.models.mlp import (
    MLP,
    build_layer_sizes,
    class_weights,
    fit_mlp_fold,
    ohe_feature_matrix,
)
from data_science_stuff.kaggle.models.realmlp import (
    REALMLP_TD_CONFIG,
    RealMLP_TD_Classifier,
)

__all__ = [
    "MLP",
    "REALMLP_TD_CONFIG",
    "RealMLP_TD_Classifier",
    "build_layer_sizes",
    "class_weights",
    "fit_mlp_fold",
    "logit_adjustment",
    "ohe_feature_matrix",
    "smooth_ce_loss",
]
