"""Reusable Kaggle competition machinery.

Extracted from the playground-series competitions (primarily s6e6); see also
:mod:`data_science_stuff.kaggle_utils` for the original per-class decision
weight tuning and CV-result logging helpers.
"""

from data_science_stuff.kaggle.blending import (
    Normalize,
    ScoreFn,
    blend,
    diversity_report,
    normalize_weights,
    optimize_blend_weights,
)
from data_science_stuff.kaggle.cv import CVResult, FitFoldFn, Matrix, run_cv
from data_science_stuff.kaggle.decision import (
    cascade_combine,
    cost_decide,
    fit_cost_matrix,
    make_cost_matrix,
    optimize_thresholds,
    split_half_gate,
)
from data_science_stuff.kaggle.device import get_lgbm_device
from data_science_stuff.kaggle.io import (
    CompetitionDirs,
    competition_dirs,
    load_params,
    save_threshold_weights,
    write_submission,
)

__all__ = [
    "CVResult",
    "CompetitionDirs",
    "FitFoldFn",
    "Matrix",
    "Normalize",
    "ScoreFn",
    "blend",
    "cascade_combine",
    "competition_dirs",
    "cost_decide",
    "diversity_report",
    "fit_cost_matrix",
    "get_lgbm_device",
    "load_params",
    "make_cost_matrix",
    "normalize_weights",
    "optimize_blend_weights",
    "optimize_thresholds",
    "run_cv",
    "save_threshold_weights",
    "split_half_gate",
    "write_submission",
]
