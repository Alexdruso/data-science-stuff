"""Train/test covariate-shift diagnostics — including the shift in the *masks*.

Two things this module exists to catch.

**1. Shift you can see in the values.** :func:`adversarial_validation` trains a
classifier to tell train rows from test rows; its AUC is the amount of shift.
Above ~0.6 the consequence is concrete: OOF stops being able to *rank* models
whose scores differ by less than the shift-induced error, even while remaining
well calibrated in absolute terms (s6e7 at adv-AUC 0.65 saw a higher-OOF
ensemble score lower on the leaderboard, repeatedly). Trust OOF for coarse
decisions, and stop believing sub-noise orderings.

**2. Shift you cannot see in the values — the mask mechanism.**
:func:`mask_mechanism_report` runs the same idea on the *missingness indicator*
matrix. s6e7's train set had four columns whose masks were deterministic
single-column trigger rules (``water_intake`` missing ⟺ ``gender == female``),
while in test the same columns were masked *uniformly at the same marginal
rates*. Every rate-level check passed; the mechanism had still flipped, trees
had learned train-only NaN↔trigger couplings that misfire on two thirds of
test NaN rows, and 1.25% of test rows carried missingness patterns that are
impossible in train. Compare per-column MAR AUCs between the splits, not
marginal null rates.

Repairing such a shift is a *surface* change, and surfaces do not mix: if you
re-mask train to emulate the test mechanism, every OOF array produced that way
lives on a different surface, reads systematically lower, and must only be
compared with — and stacked against — other arrays from the same surface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#: Defaults for the shift classifiers: enough capacity to find a real shift,
#: cheap enough to run per column.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def _default_estimator() -> Any:
    from lightgbm import LGBMClassifier  # noqa: PLC0415

    return LGBMClassifier(**DEFAULT_PARAMS)


@dataclass
class AdversarialResult:
    """Result of :func:`adversarial_validation`."""

    auc: float
    """OOF AUC of "is this row from test?". 0.5 = no detectable shift."""
    train_p_test: NDArray[np.float64]
    """Per-train-row P(test), in the input row order."""
    test_p_test: NDArray[np.float64]
    """Per-test-row P(test), in the input row order."""


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cat_cols: Sequence[str] = (),
    n_splits: int = 5,
    seed: int = 42,
    estimator_factory: Callable[[], Any] | None = None,
) -> AdversarialResult:
    """Cross-validated "is this row from test?" classifier over the shared columns.

    Args:
        train: Train feature frame.
        test: Test feature frame.
        cat_cols: Columns to cast to pandas ``category`` (cast on the pooled
            frame so both splits share a category set).
        n_splits: Stratified folds.
        seed: Fold and estimator seed.
        estimator_factory: Zero-argument factory returning a fresh unfitted
            sklearn-style classifier; default is LightGBM with
            :data:`DEFAULT_PARAMS`.

    Returns:
        An :class:`AdversarialResult` — the AUC plus the per-row P(test), which
        doubles as importance weights via :func:`importance_weights`.
    """
    shared = [c for c in train.columns if c in test.columns]
    if not shared:
        msg = "train and test share no columns"
        raise ValueError(msg)
    pooled = pd.concat([train[shared], test[shared]], ignore_index=True)
    for col in cat_cols:
        if col in pooled.columns:
            pooled[col] = pooled[col].astype("category")
    is_test = np.concatenate([np.zeros(len(train)), np.ones(len(test))]).astype(
        np.int64
    )

    make = _default_estimator if estimator_factory is None else estimator_factory
    oof = np.zeros(len(pooled))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, val_idx in folds.split(pooled, is_test):
        model = make()
        model.fit(pooled.iloc[fit_idx], is_test[fit_idx])
        oof[val_idx] = model.predict_proba(pooled.iloc[val_idx])[:, 1]
    return AdversarialResult(
        auc=float(roc_auc_score(is_test, oof)),
        train_p_test=oof[: len(train)],
        test_p_test=oof[len(train) :],
    )


def mask_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness indicator matrix: one boolean column per input column.

    >>> import pandas as pd
    >>> mask_frame(pd.DataFrame({"a": [1.0, None]}))["a"].tolist()
    [False, True]
    """
    return df.isna()


def mask_mechanism_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    n_splits: int = 3,
    seed: int = 42,
    estimator_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Per-column MAR test: is a column's mask predictable from the other masks?

    For each column, fits "is this column missing?" against the *other* columns'
    missingness indicators, separately within train and within test. An AUC
    near 0.5 means the mask is MCAR-like within that split; near 1 means the
    mask follows a rule. **A large train-vs-test AUC gap is a mechanism shift
    even when the marginal null rates match exactly** — which is precisely the
    case that rate-level EDA cannot see.

    Args:
        train: Train frame (features only).
        test: Test frame (features only).
        columns: Columns to test; default every shared column with missing
            values in either split.
        n_splits: Stratified folds per column per split.
        seed: Fold and estimator seed.
        estimator_factory: See :func:`adversarial_validation`.

    Returns:
        One row per column with ``train_rate``, ``test_rate``, ``train_auc``,
        ``test_auc`` and ``auc_gap`` (``|train_auc - test_auc|``), sorted by
        ``auc_gap`` descending. AUCs are NaN where a split has too few missing
        rows to cross-validate.
    """
    train_mask, test_mask = mask_frame(train), mask_frame(test)
    shared = [c for c in train_mask.columns if c in test_mask.columns]
    if columns is None:
        columns = [
            c for c in shared if bool(train_mask[c].any()) or bool(test_mask[c].any())
        ]
    make = _default_estimator if estimator_factory is None else estimator_factory

    def mar_auc(mask: pd.DataFrame, col: str) -> float:
        target = mask[col].to_numpy().astype(np.int64)
        others = mask[[c for c in shared if c != col]]
        counts = np.bincount(target, minlength=2)
        if others.shape[1] == 0 or counts.min() < n_splits:
            return float("nan")
        oof = np.zeros(len(target))
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, val_idx in folds.split(others, target):
            model = make()
            model.fit(others.iloc[fit_idx], target[fit_idx])
            oof[val_idx] = model.predict_proba(others.iloc[val_idx])[:, 1]
        return float(roc_auc_score(target, oof))

    rows = [
        {
            "column": col,
            "train_rate": float(train_mask[col].mean()),
            "test_rate": float(test_mask[col].mean()),
            "train_auc": mar_auc(train_mask, col),
            "test_auc": mar_auc(test_mask, col),
        }
        for col in columns
    ]
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["auc_gap"] = (report["train_auc"] - report["test_auc"]).abs()
    return report.sort_values("auc_gap", ascending=False).reset_index(drop=True)


def unseen_mask_patterns(train: pd.DataFrame, test: pd.DataFrame) -> NDArray[np.bool_]:
    """Test rows whose missingness *pattern* never occurs in train.

    A pattern impossible in train (s6e7: two columns co-missing, which the
    train trigger rules made a contradiction) is a row no model has ever been
    fitted on. Counting them prices the mechanism shift's tail directly.

    Returns:
        Boolean array over test rows, ``True`` where the pattern is unseen.

    >>> import pandas as pd
    >>> tr = pd.DataFrame({"a": [1.0, None], "b": [1.0, 1.0]})
    >>> te = pd.DataFrame({"a": [1.0, None], "b": [None, 1.0]})
    >>> unseen_mask_patterns(tr, te).tolist()
    [True, False]
    """
    shared = [c for c in train.columns if c in test.columns]
    train_pat = mask_frame(train[shared]).apply(
        lambda row: tuple(row.to_numpy().tolist()), axis=1
    )
    test_pat = mask_frame(test[shared]).apply(
        lambda row: tuple(row.to_numpy().tolist()), axis=1
    )
    seen = set(train_pat)
    return np.asarray([pat not in seen for pat in test_pat], dtype=np.bool_)


def importance_weights(
    p_test: NDArray[np.float64], *, clip: float = 10.0
) -> NDArray[np.float64]:
    """Importance weights ``p / (1 - p)`` from adversarial P(test), clipped.

    Upweights the train rows that look most like test, giving an offline metric
    aimed at the distribution you are actually scored on. Clip: the ratio is
    unbounded, and a handful of near-1 rows would otherwise carry the metric.

    >>> import numpy as np
    >>> importance_weights(np.array([0.5])).tolist()
    [1.0]
    """
    ratio = p_test / np.clip(1.0 - p_test, 1e-12, None)
    return np.asarray(np.clip(ratio, 0.0, clip), dtype=np.float64)


def weighted_balanced_accuracy(
    y: NDArray[np.int64],
    pred: NDArray[np.int64],
    sample_weight: NDArray[np.float64],
) -> float:
    """Balanced accuracy with per-sample weights (weighted mean of per-class recalls).

    Pair with :func:`importance_weights` to score on the test-like region.
    Classes absent from ``y`` are skipped.

    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> weighted_balanced_accuracy(y, np.array([0, 1, 1, 1]), np.ones(4))
    0.75
    """
    recalls = [
        float(np.average(pred[y == c] == c, weights=sample_weight[y == c]))
        for c in np.unique(y)
    ]
    return float(np.mean(recalls))


def uniform_remask(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    *,
    mult: float = 1.0,
    seed: int = 20260707,
) -> pd.DataFrame:
    """Re-mask train so its missingness *mechanism* matches a uniform test mask.

    The s6e7 repair: when :func:`mask_mechanism_report` shows train NaNs
    follow deterministic rules while test NaNs are uniform at the same rate,
    hide observed train values at random, at each column's test NaN rate
    times ``mult``. That breaks the train-only coupling between "is NaN" and
    the rule that produced it. Test is never re-masked.

    Two things to keep in mind:

    - The seed is fixed on purpose: every model and every seed must see the
      same mask, or their OOF arrays are not blendable.
    - The result is a different *surface* (root CLAUDE.md, lesson 9): OOF
      from re-masked train reads lower by construction and may only be
      compared with or stacked against OOF from the same re-masked train.

    Args:
        train: Training features.
        test: Test features (source of the per-column NaN rates).
        columns: Columns whose mechanism differs; others are left alone.
            Columns absent from ``train`` are skipped.
        mult: Scales the re-mask rate. At 1.0 the train NaN volume roughly
            doubles in those columns (rule NaNs plus random ones); below 1
            trades weaker decoupling for a closer volume match.
        seed: RNG seed for the mask.

    Returns:
        A re-masked copy of ``train``.
    """
    rng = np.random.default_rng(seed)
    out = train.copy()
    for col in columns:
        if col not in out.columns:
            continue
        rate = float(test[col].isna().mean()) * mult
        hit = out[col].notna().to_numpy() & (rng.random(len(out)) < rate)
        out.loc[out.index[hit], col] = np.nan
    return out
