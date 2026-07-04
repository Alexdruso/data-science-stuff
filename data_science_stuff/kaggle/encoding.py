"""Leakage-safe categorical encoding helpers.

Generalized from the s6e6 cdeotte-recipe ports (``train_xgb_deotte.py`` /
``deotte_features.py``): fold-safe per-class target encoding, deterministic
factorization shared across splits, quantile-bin categorical construction,
and frequency features.

The golden rule these helpers enforce: anything fit from targets or from
distributional statistics is fit on the *training* rows only (a fold's
training split, or an explicit ``fit_mask``), then transformed onto
validation/test. Fitting globally leaks validation targets and inflates OOF
scores that do not transfer to the leaderboard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import TargetEncoder


def cat_key(s: pd.Series) -> pd.Series:
    """String key for a categorical column, with missing values as ``__NA__``."""
    return s.fillna("__NA__").astype(str)


def sorted_factorize(*series: pd.Series) -> tuple[NDArray[np.int32], ...]:
    """Deterministic integer codes shared across any number of splits.

    The category-to-code mapping is built from the *union* of all inputs
    (sorted lexicographically), so the same category gets the same code in
    train/val/test. Values missing from the union map to -1 (defensive; the
    union construction means this does not normally happen).

    Returns:
        One int32 code array per input series, in input order.
    """
    keyed = [cat_key(s) for s in series]
    union = pd.concat(keyed, ignore_index=True)
    mapping = {category: code for code, category in enumerate(sorted(union.unique()))}
    return tuple(np.asarray(k.map(mapping).fillna(-1), dtype=np.int32) for k in keyed)


def te_source_columns(
    selected_features: Sequence[str],
    cat_cols: Sequence[str],
    *,
    prefix: str = "TE_",
) -> list[str]:
    """Categorical columns whose target encodings appear in a feature list.

    Given a curated feature list containing names like ``TE_<col>_<class>``,
    return the categorical source columns that actually need encoding —
    avoids computing expensive encodings that the model never sees.
    """
    return [
        c
        for c in cat_cols
        if any(f.startswith(f"{prefix}{c}_") for f in selected_features)
    ]


def add_fold_safe_target_encoding(
    X_tr: pd.DataFrame,
    y_tr: NDArray[np.int64],
    others: Sequence[pd.DataFrame],
    te_cols: Sequence[str],
    class_names: Mapping[int, str],
    *,
    smooth: float = 20.0,
    inner_cv: int = 5,
    seed: int = 219,
    name_fmt: str = "TE_{col}_{cls}",
) -> tuple[pd.DataFrame, ...]:
    """Per-class binary target encoding, fit on the fold's training split only.

    For every column in ``te_cols`` and every class, fits a one-vs-rest
    :class:`sklearn.preprocessing.TargetEncoder` (``target_type="binary"``,
    nested ``cv=inner_cv`` for honest in-fold encoding) on ``(X_tr, y_tr)``
    and transforms the training frame plus every frame in ``others``
    (validation, test, ...). Call this INSIDE the fold loop — never on the
    full training set.

    Args:
        X_tr: The fold's training features (not mutated; a copy is returned).
        y_tr: The fold's training labels, encoded 0..n_classes-1.
        others: Frames to transform with the fitted encoders (val, test, ...).
        te_cols: Categorical source columns to encode (missing ones skipped).
        class_names: ``{class_index: class_name}`` used in feature names.
        smooth: TargetEncoder smoothing strength.
        inner_cv: Nested CV splits inside ``fit_transform`` (the fold-safe part).
        seed: ``random_state`` for the encoder's internal shuffling.
        name_fmt: Feature-name template with ``{col}`` and ``{cls}`` fields.

    Returns:
        ``(X_tr_encoded, *others_encoded)`` — copies with the TE columns added.
    """
    x_tr = X_tr.copy()
    frames = [frame.copy() for frame in others]
    for col in te_cols:
        if col not in x_tr.columns:
            continue
        key_tr = cat_key(x_tr[col]).to_frame(col)
        keys = [cat_key(frame[col]).to_frame(col) for frame in frames]
        for cls_idx, cls_name in class_names.items():
            y_bin = (y_tr == cls_idx).astype(np.int32)
            encoder = TargetEncoder(
                target_type="binary",
                smooth=smooth,
                cv=inner_cv,
                shuffle=True,
                random_state=seed,
            )
            name = name_fmt.format(col=col, cls=cls_name)
            x_tr[name] = encoder.fit_transform(key_tr, y_bin).ravel().astype(np.float32)
            for frame, key in zip(frames, keys):
                frame[name] = encoder.transform(key).ravel().astype(np.float32)
    return (x_tr, *frames)


def qcut_codes(
    values: NDArray[Any], ref_values: NDArray[Any], q: int
) -> NDArray[np.int16]:
    """Quantile-bin codes with edges computed from ``ref_values`` only.

    Bin edges come from the reference distribution (typically the training
    rows), so binning test values cannot leak test statistics into features.
    Out-of-range and NaN values code to -1; degenerate references (fewer than
    two finite values, or a constant column) code everything to -1.
    """
    ref = ref_values[~np.isnan(ref_values)]
    if len(ref) < 2:  # noqa: PLR2004 — need two points to form one bin edge pair
        return np.full(len(values), -1, dtype=np.int16)
    bins = np.unique(np.quantile(ref, np.linspace(0, 1, q + 1)).astype(np.float32))
    if len(bins) <= 1:
        return np.full(len(values), -1, dtype=np.int16)
    codes = np.searchsorted(bins, values, side="left") - 1
    codes = np.where(values == bins[0], 0, codes)
    codes = np.where(
        (values < bins[0]) | (values > bins[-1]) | np.isnan(values), -1, codes
    )
    return np.clip(codes, -1, len(bins) - 2).astype(np.int16)


def add_quantile_bin_features(
    df: pd.DataFrame,
    fit_mask: NDArray[np.bool_],
    *,
    cols: Sequence[str],
    quantiles: Sequence[int] = (16, 64, 256),
    crosses: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Quantile-bin categorical features (plus optional bin crosses).

    Each numeric column in ``cols`` becomes ``<col>_qbin<q>`` string
    categoricals for every ``q``, with bin edges fit on ``fit_mask`` rows
    only (pass the train-rows mask of a concatenated train/test frame).
    ``crosses`` are pairs of *already-created* qbin column names to combine
    into ``<a>__x__<b>`` interaction categoricals — the recipe that feeds
    :func:`add_fold_safe_target_encoding`.

    Returns:
        ``(frame_with_features, new_column_names)``.
    """
    out = df.copy()
    qbin_cols: list[str] = []
    for col in cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float32)
        ref = series[fit_mask]
        for q in quantiles:
            name = f"{col}_qbin{q}"
            out[name] = qcut_codes(series, ref, q).astype(np.int16).astype(str)
            qbin_cols.append(name)
    for a, b in crosses:
        if a in out.columns and b in out.columns:
            name = f"{a}__x__{b}"
            out[name] = cat_key(out[a]) + "__" + cat_key(out[b])
            qbin_cols.append(name)
    return out, qbin_cols


def select_low_cardinality_cols(
    df: pd.DataFrame, cat_cols: Sequence[str], max_card: int
) -> list[str]:
    """Categorical columns whose cardinality stays at or under ``max_card``.

    High-cardinality categoricals make target encodings noisy and slow;
    cap them (s6e6 used ``max_card=5000``).
    """
    return [
        c
        for c in cat_cols
        if c in df.columns and int(cat_key(df[c]).nunique(dropna=False)) <= max_card
    ]


def add_frequency_features(
    df: pd.DataFrame, cols: Sequence[str], fit_mask: NDArray[np.bool_]
) -> pd.DataFrame:
    """Category frequency (+ log1p) features counted on ``fit_mask`` rows only.

    Frequencies are distributional statistics: counting over test rows would
    leak test-set composition, so counts come from the masked (train) rows
    and unseen categories map to 0.
    """
    out: pd.DataFrame = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        key = cat_key(out[col])
        counts = key[fit_mask].value_counts(dropna=False)
        freq = key.map(counts).fillna(0).astype("float32")
        out[f"{col}_freq"] = freq
        out[f"{col}_freq_log1p"] = np.log1p(freq.to_numpy(np.float32)).astype("float32")
    return out
