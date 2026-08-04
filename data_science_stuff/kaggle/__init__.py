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
    bagged_decision_weights,
    balanced_accuracy_fast,
    cascade_combine,
    cost_decide,
    fit_cost_matrix,
    make_cost_matrix,
    optimize_thresholds,
    split_half_gate,
    tune_decision_weights_grid,
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
from data_science_stuff.kaggle.screening import (
    DecisionLayer,
    MechanismScreen,
    argmax_layer,
    calibrate_lambda,
    classify_exponent,
    decay_exponent,
    frozen_weight_layer,
    honest_weighted_layer,
    mean_shift_control,
    screen_mechanism,
    selection_null_arms,
    variance_reduction_control,
)
from data_science_stuff.kaggle.shift import (
    AdversarialResult,
    adversarial_validation,
    importance_weights,
    mask_frame,
    mask_mechanism_report,
    unseen_mask_patterns,
    weighted_balanced_accuracy,
)
from data_science_stuff.kaggle.stacking import (
    CaruanaResult,
    caruana_select,
    clipped_logit,
    stack_oof,
)

__all__ = [
    "AdversarialResult",
    "CVResult",
    "CaruanaResult",
    "CompetitionDirs",
    "DecisionLayer",
    "FitFoldFn",
    "Matrix",
    "MechanismScreen",
    "Normalize",
    "ScoreFn",
    "add_fold_safe_target_encoding",
    "add_frequency_features",
    "add_quantile_bin_features",
    "adversarial_validation",
    "argmax_layer",
    "bagged_decision_weights",
    "balanced_accuracy_fast",
    "blend",
    "calibrate_lambda",
    "caruana_select",
    "cascade_combine",
    "cat_key",
    "classify_exponent",
    "clipped_logit",
    "competition_dirs",
    "cost_decide",
    "decay_exponent",
    "diversity_report",
    "fit_cost_matrix",
    "frozen_weight_layer",
    "get_lgbm_device",
    "honest_weighted_layer",
    "importance_weights",
    "load_params",
    "make_cost_matrix",
    "mask_frame",
    "mask_mechanism_report",
    "mean_shift_control",
    "normalize_weights",
    "optimize_blend_weights",
    "optimize_thresholds",
    "qcut_codes",
    "run_cv",
    "save_threshold_weights",
    "screen_mechanism",
    "select_low_cardinality_cols",
    "selection_null_arms",
    "sorted_factorize",
    "split_half_gate",
    "stack_oof",
    "te_source_columns",
    "tune_decision_weights_grid",
    "unseen_mask_patterns",
    "variance_reduction_control",
    "weighted_balanced_accuracy",
    "write_submission",
]
