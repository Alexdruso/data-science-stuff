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
from data_science_stuff.kaggle.encoding import (
    add_fold_safe_target_encoding,
    add_frequency_features,
    add_quantile_bin_features,
    cat_key,
    qcut_codes,
    select_low_cardinality_cols,
    sorted_factorize,
    te_source_columns,
)
from data_science_stuff.kaggle.io import (
    CompetitionDirs,
    competition_dirs,
    load_params,
    save_threshold_weights,
    write_submission,
)
from data_science_stuff.kaggle.stacking import (
    CaruanaResult,
    caruana_select,
    clipped_logit,
    stack_oof,
)

__all__ = [
    "CVResult",
    "CaruanaResult",
    "CompetitionDirs",
    "FitFoldFn",
    "Matrix",
    "Normalize",
    "ScoreFn",
    "add_fold_safe_target_encoding",
    "add_frequency_features",
    "add_quantile_bin_features",
    "blend",
    "caruana_select",
    "cascade_combine",
    "cat_key",
    "clipped_logit",
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
    "qcut_codes",
    "run_cv",
    "save_threshold_weights",
    "select_low_cardinality_cols",
    "sorted_factorize",
    "split_half_gate",
    "stack_oof",
    "te_source_columns",
    "write_submission",
]
