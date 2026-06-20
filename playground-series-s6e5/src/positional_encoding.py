"""NeRF-style sinusoidal positional encoding for tabular features.

Deterministic frequency lifting: γ(x) = [sin(π·2^k · x), cos(π·2^k · x)] for k=0..L-1
Applied to a curated subset of positional/ordinal numeric features where non-linear
and high-frequency patterns are expected (TyreLife degradation curves, lap windows, etc.).

Two entry points:
  apply_fourier_np  — for MLP/GRU (numpy arrays, applied after Yeo-Johnson scaling)
  apply_fourier_df  — for LGBM/XGB/CatBoost (pandas DataFrames, applied after per-fold MinMaxScaler)
"""

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PE_FEATURE_NAMES: list[str] = [
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "Stint",
    "TyreLife_frac",
    "window_gap",
    "laps_past_window",
]


def apply_fourier_np(
    X: np.ndarray,
    encode_indices: list[int],
    L: int,
) -> np.ndarray:
    """Append sin/cos PE columns to a float32 numpy array (in-place concatenation).

    Inputs are expected to already be scaled (e.g. Yeo-Johnson + StandardScaler).
    Returns shape (n, d + 2*L*len(encode_indices)).
    """
    if L == 0 or not encode_indices:
        return X
    freqs = np.array([math.pi * (2.0**k) for k in range(L)], dtype=np.float32)
    selected = X[:, encode_indices]          # (n, n_enc)
    args = selected[:, :, None] * freqs      # (n, n_enc, L)  broadcast
    sins = np.sin(args).reshape(len(X), -1)  # (n, n_enc*L)
    coss = np.cos(args).reshape(len(X), -1)
    return np.concatenate([X, sins, coss], axis=1)


def apply_fourier_df(
    df: pd.DataFrame,
    col_names: list[str],
    L: int,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Append sin/cos PE columns to a pandas DataFrame (for tree models).

    col_names must be a subset of df.columns.
    scaler must already be fit on the training fold (call MinMaxScaler().fit(...) externally).
    New columns are named pe_sin_{k}_{col} and pe_cos_{k}_{col}.
    """
    if L == 0 or not col_names:
        return df
    present = [c for c in col_names if c in df.columns]
    x_norm = scaler.transform(df[present]).astype(np.float64)   # (n, n_enc) in [0,1]
    freqs = np.array([math.pi * (2.0**k) for k in range(L)], dtype=np.float64)
    args = x_norm[:, :, None] * freqs   # (n, n_enc, L)
    sins = np.sin(args).reshape(len(df), -1)
    coss = np.cos(args).reshape(len(df), -1)
    sin_cols = [f"pe_sin_{k}_{c}" for k in range(L) for c in present]
    cos_cols = [f"pe_cos_{k}_{c}" for k in range(L) for c in present]
    result = df.copy()
    result[sin_cols] = sins
    result[cos_cols] = coss
    return result
