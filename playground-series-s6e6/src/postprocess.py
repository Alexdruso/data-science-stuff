"""Post-processing utilities for PS S6E6 training scripts — package re-export.

The implementations were extracted to the shared package
(`data_science_stuff.kaggle.decision.optimize_thresholds`, a thin wrapper
over `kaggle_utils.tune_decision_weights`, and
`data_science_stuff.kaggle.io.save_threshold_weights`). This shim keeps the
~50 existing `from postprocess import ...` call sites working unchanged —
extend the package, not this file.

Behavior notes versus the pre-extraction implementation (results are
decision-rule-equivalent, not bit-identical): weights come back
max-normalized (argmax(p*w) is scale-invariant, so predictions are
unchanged), the optimizer fixes class 0's log-weight at 0 (K-1 free DOF),
and the restart RNG stream differs (differences within optimizer noise).
"""

from data_science_stuff.kaggle.decision import optimize_thresholds
from data_science_stuff.kaggle.io import save_threshold_weights

__all__ = ["optimize_thresholds", "save_threshold_weights"]
