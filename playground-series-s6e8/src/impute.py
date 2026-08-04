"""Fold-aware null-handling strategies for the S6E8 arm sweep.

Every transform here has the signature
``(x_tr, y_tr, x_va, x_te) -> (x_tr, x_va, x_te)`` and is called from *inside*
`fit_fold`, so all statistics (medians, modes, regression coefficients) are fit
on the fold's training split only. Fitting them on the full matrix would leak
val rows into the imputer — CLAUDE.md leakage rule #2, and with 4-20% missing
in every column that is a real surface here, not a formality.

Which columns are worth imputing was decided from the data, not by default:

    column                     single-feature AUC   OLS R^2 from siblings
    daily_screen_time_hours          0.889                0.807
    weekend_screen_time              0.880                0.640
    social_media_hours               0.858                0.482
    work_study_hours                 0.653                0.419
    gaming_hours                     0.621                0.312
    age / sleep / notifications / app_opens
                                  0.49 - 0.54           0.000 - 0.005

The bottom group is both unpredictable from its siblings AND uninformative
about the target, so imputing it can only add noise. Only the top group is
imputed by `relational_impute`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from features import RAW_CATEGORICAL_COLS, RAW_NUMERIC_COLS
from features_v2 import BUDGET_PARTS, derive_v2

RAW_COLS: list[str] = [*RAW_NUMERIC_COLS, *RAW_CATEGORICAL_COLS]

#: Chained imputation order: the three non-negative components of `daily`
#: first, then `daily` itself (so the budget clip has complete parts to use),
#: then `weekend` (which `daily` predicts best).
IMPUTE_ORDER: list[str] = [
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "daily_screen_time_hours",
    "weekend_screen_time",
]

#: Never imputed — R^2 ~ 0 from siblings and AUC ~ 0.50 against the target.
NEVER_IMPUTE: list[str] = [
    "age",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
]

Frames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


# ────────────────────────────── masks ──────────────────────────────


def _with_masks(x: pd.DataFrame) -> pd.DataFrame:
    """Add one `isnull_<col>` per raw column plus `n_missing`.

    Marginally these are worthless here (every `is_null(col)` has AUC
    0.4999-0.5010 against the target). They are carried anyway so that the
    imputation arms are compared against a control that also has them: without
    that, an imputation arm's delta would confound "imputing helps" with
    "losing the NaN signal hurts".
    """
    out = x.copy()
    masks = {f"isnull_{c}": x[c].isna().astype(np.int8) for c in RAW_COLS}
    masks["n_missing"] = (
        pd.DataFrame(masks).sum(axis=1).astype(np.int8)  # type: ignore[assignment]
    )
    return pd.concat([out, pd.DataFrame(masks, index=x.index)], axis=1)


def add_masks(
    x_tr: pd.DataFrame, _y_tr: np.ndarray, x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s2`: masks only, NaN left native. Row-local, nothing is fit."""
    return _with_masks(x_tr), _with_masks(x_va), _with_masks(x_te)


# ─────────────────────── derived-feature refresh ───────────────────────


def _recompute_derived(x: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every derived column from the (now imputed) raw columns.

    This is where most of the value of imputing lives: the ratio features are
    NaN whenever *either* operand is NaN, and screen-block nulls co-occur
    (lift ~2.2), so a large share of rows lose them. Reuses `derive_v2` rather
    than restating the formulas, and `derive_v2` does not sort — the fold row
    order is preserved.
    """
    raw = x[RAW_COLS].copy()
    for col in RAW_CATEGORICAL_COLS:
        raw[col] = raw[col].astype(object)
    rebuilt = derive_v2(pl.from_pandas(raw))
    derived = [c for c in rebuilt.columns if c not in RAW_COLS]
    new = rebuilt.select(derived).to_pandas()
    new.index = x.index
    out = x.copy()
    for col in derived:
        if col in out.columns:
            out[col] = new[col]
    return out


# ───────────────────────── simple imputation ─────────────────────────


def median_impute(
    x_tr: pd.DataFrame, _y_tr: np.ndarray, x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s3`: fold-fit median (numeric) / mode (categorical), plus masks.

    The naive strategy, included as the honest control for `relational_impute`
    — if the smart imputer cannot beat this, its extra machinery is not the
    thing doing the work.
    """
    medians = {c: float(x_tr[c].median()) for c in RAW_NUMERIC_COLS}
    modes = {
        c: x_tr[c].mode(dropna=True) for c in RAW_CATEGORICAL_COLS
    }  # empty Series if all-null

    def apply(x: pd.DataFrame) -> pd.DataFrame:
        out = _with_masks(x)
        for col, val in medians.items():
            out[col] = out[col].fillna(val)
        for col, mode in modes.items():
            if len(mode):
                out[col] = out[col].fillna(mode.iloc[0])
        return _recompute_derived(out)

    return apply(x_tr), apply(x_va), apply(x_te)


# ─────────────────── constraint-aware relational imputation ───────────────────


def _lgbm_regressor() -> object:
    from lightgbm import LGBMRegressor  # noqa: PLC0415

    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=12,
    )


def _project_parts_onto_budget(
    frame: pd.DataFrame, was_null: dict[str, np.ndarray]
) -> None:
    """Scale imputed parts down so `social + gaming + work <= daily`, in place.

    The clip inside `impute_column` only floors an *imputed* `daily` at its
    parts. The other direction also has to hold: on a row where `daily` is
    observed, freely-predicted parts can overshoot it — the first smoke test
    produced `min(daily - sgw_lb) = -0.43`. Observed parts are never touched
    (they already satisfy the constraint); only the imputed ones absorb the
    correction, proportionally.
    """
    daily = frame["daily_screen_time_hours"].to_numpy(dtype=np.float64)
    parts = frame[BUDGET_PARTS].to_numpy(dtype=np.float64)
    adjustable = np.column_stack([was_null[c] for c in BUDGET_PARTS])

    total = np.nansum(parts, axis=1)
    over = np.isfinite(daily) & (total > daily + 1e-9)
    if not over.any():
        return

    imputed_mass = np.where(adjustable, np.nan_to_num(parts), 0.0).sum(axis=1)
    observed_mass = total - imputed_mass
    budget = np.clip(daily - observed_mass, 0.0, None)
    scale = np.divide(
        budget,
        imputed_mass,
        out=np.zeros_like(budget),
        where=imputed_mass > 1e-12,
    )
    for j, col in enumerate(BUDGET_PARTS):
        sel = over & adjustable[:, j]
        if sel.any():
            values = parts[:, j].copy()
            values[sel] = values[sel] * scale[sel]
            frame[col] = values


def relational_impute(
    x_tr: pd.DataFrame, _y_tr: np.ndarray, x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s4`: chained sibling regression + the verified budget clip, plus masks.

    For each column in `IMPUTE_ORDER`, a LightGBM regressor is fit on the
    training-fold rows where that column is *observed*, using the other raw
    numerics and their masks as predictors, and used to fill the rows where it
    is missing. Predictors are median-filled first so the regressor sees a
    dense matrix; the masks tell it which values were real.

    The distinctive part is the clip. Because
    `social + gaming + work <= daily` holds in 100.0000% of complete rows and
    every term is non-negative, an imputed `daily` can be floored at the sum of
    its parts. That is accuracy no generic imputer gets, and it is the content
    of this arm over `median_impute`.
    """
    predictors = list(RAW_NUMERIC_COLS)
    fill = {c: float(x_tr[c].median()) for c in predictors}

    frames = [_with_masks(f) for f in (x_tr, x_va, x_te)]
    # Snapshot of what was missing, so later chained steps still fill only the
    # originally-missing cells and never overwrite an observed value.
    was_null = [{c: f[c].isna().to_numpy() for c in IMPUTE_ORDER} for f in frames]

    def impute_column(target: str) -> None:
        feats = [c for c in predictors if c != target]
        mask_feats = [f"isnull_{c}" for c in feats]

        def design(frame: pd.DataFrame) -> pd.DataFrame:
            block = frame[feats].copy()
            for col in feats:
                block[col] = block[col].fillna(fill[col])
            return pd.concat([block, frame[mask_feats]], axis=1)

        observed = ~was_null[0][target]
        if observed.sum() < 1000 or (~observed).sum() == 0:
            return
        model = _lgbm_regressor()
        model.fit(design(frames[0]).loc[observed], frames[0].loc[observed, target])

        for frame, nulls in zip(frames, was_null):
            idx = nulls[target]
            if not idx.any():
                continue
            pred = np.asarray(model.predict(design(frame).loc[idx]), dtype=np.float64)
            if target == "daily_screen_time_hours":
                # Floor an imputed total at the sum of its parts.
                lower = frame.loc[idx, BUDGET_PARTS].sum(axis=1, skipna=True).to_numpy()
                pred = np.maximum(pred, lower)
            frame.loc[idx, target] = np.clip(pred, 0.0, None)

    # Phase A — the three non-negative components.
    for target in BUDGET_PARTS:
        impute_column(target)
    # Phase B — enforce the budget from ABOVE, where `daily` is observed.
    for frame, nulls in zip(frames, was_null):
        _project_parts_onto_budget(frame, nulls)
    # Phase C/D — the total (floored by its now-consistent parts), then weekend.
    impute_column("daily_screen_time_hours")
    impute_column("weekend_screen_time")

    # Anything still missing (the never-impute group, and the categoricals) is
    # left as NaN on purpose — LightGBM handles it natively and there is
    # nothing to reconstruct it from.
    return (
        _recompute_derived(frames[0]),
        _recompute_derived(frames[1]),
        _recompute_derived(frames[2]),
    )


# ───────────────────── pooled latent screen intensity ─────────────────────


def latent_impute(
    x_tr: pd.DataFrame, _y_tr: np.ndarray, x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s5`: impute the *latent*, not the columns.

    `daily`, `social`, `weekend` are three noisy views of one quantity
    (pairwise r 0.48-0.80, single-feature AUC 0.86-0.89). Rather than
    reconstructing each column, estimate one standardised "screen intensity"
    score from whichever views the row actually has, and hand the tree that
    plus how many views it rests on. Differs in kind from `s4`, which is why
    it is a separate arm rather than a variant.
    """
    views = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"]
    stats = {c: (float(x_tr[c].mean()), float(x_tr[c].std()) or 1.0) for c in views}

    def apply(x: pd.DataFrame) -> pd.DataFrame:
        out = _with_masks(x)
        z = pd.DataFrame(
            {c: (x[c] - stats[c][0]) / stats[c][1] for c in views}, index=x.index
        )
        out["screen_latent"] = z.mean(axis=1, skipna=True)
        out["screen_latent_n"] = z.notna().sum(axis=1).astype(np.int8)
        out["screen_latent_sd"] = z.std(axis=1, skipna=True)
        return out

    return apply(x_tr), apply(x_va), apply(x_te)
