"""Faithful CPU/pandas port of cdeotte's XGB v1 feature recipe (S6E6, CV 0.96694).

The original is RAPIDS (cuDF/cuML/cuPy) Kaggle code. This reproduces the SAME feature
set in pandas + numpy so it runs in our env (xgboost-CUDA for the model). The per-class
fold-safe TargetEncoder on quantile bins is done in the TRAINING loop (see train_xgb_deotte),
not here, because it needs the fold's y. Everything here is target-free except the original-
SDSS17 priors (which use only the EXTERNAL original's labels — leak-safe, like our
add_sdss17_priors).

Recipe pieces (cdeotte): 10 color pairs, mag/flux aggregates, redshift×band + band/redshift,
sky-coordinate trig, SED slope/curvatures, recomputed spectral_type/galaxy_population from
photometry + crosses, quantile bins (16/64/256) + 3 qbin crosses, frequency features, and
original-prior features on the binned categoricals. The model then selects ~240 TOP_FEATURES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Generic encoding helpers were extracted to the shared package; the
# re-exports below keep every existing `from deotte_features import ...`
# call site working. Only the SDSS-specific recipe stays local.
from data_science_stuff.kaggle.encoding import (
    add_frequency_features,
    cat_key,
    qcut_codes,
    select_low_cardinality_cols,
)
from data_science_stuff.kaggle.encoding import (
    add_quantile_bin_features as _add_quantile_bin_features,
)

TARGET = "class"
ID_COL = "id"
EPS = 1e-6
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]
TE_SMOOTH = 20.0
TE_INNER_SPLITS = 5
TE_MAX_CARDINALITY = 5000


def spectral_type(g: pd.Series, r: pd.Series) -> pd.Series:
    return pd.cut(r - g, [-np.inf, -1, -0.5, 0, np.inf],
                 labels=["M", "G/K", "A/F", "O/B"]).astype("str")


def galaxy_population(u: pd.Series, r: pd.Series) -> pd.Series:
    return pd.cut(u - r, [-np.inf, 2.2, np.inf],
                 labels=["Blue_Cloud", "Red_Sequence"]).astype("str")


def add_public_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RAW_NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    color_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"),
                   ("u", "i"), ("u", "z"), ("g", "i"), ("g", "z"), ("r", "z")]
    for a, b in color_pairs:
        out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    bv = out[BANDS].to_numpy(dtype=np.float32)
    out["mag_mean"] = bv.mean(axis=1)
    out["mag_std"] = bv.std(axis=1, ddof=1)
    out["mag_min"] = bv.min(axis=1)
    out["mag_max"] = bv.max(axis=1)
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")
    out["mag_argmin"] = bv.argmin(axis=1).astype(np.int16)
    out["mag_argmax"] = bv.argmax(axis=1).astype(np.int16)

    for b in BANDS:
        out[f"redshift_{b}"] = (out["redshift"] * out[b]).astype("float32")
        out[f"{b}_over_redshift"] = (out[b] / (out["redshift"].abs() + EPS)).astype("float32")

    alpha_rad = out["alpha"].to_numpy(np.float32) * np.float32(np.pi / 180.0)
    delta_rad = out["delta"].to_numpy(np.float32) * np.float32(np.pi / 180.0)
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    out["sky_x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    out["sky_y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    out["sky_z"] = np.sin(delta_rad)

    flux_cols = []
    for b in BANDS:
        clipped = np.clip(out[b].to_numpy(np.float32), -30, 30)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype(np.float32)
        out[f"flux_{b}"] = flux
        flux_cols.append(flux)
    fv = np.vstack(flux_cols).T
    out["flux_mean"] = fv.mean(axis=1)
    out["flux_std"] = fv.std(axis=1, ddof=1)
    out["flux_min"] = fv.min(axis=1)
    out["flux_max"] = fv.max(axis=1)
    out["flux_range"] = (out["flux_max"] - out["flux_min"]).astype("float32")

    x = np.arange(len(BANDS), dtype=np.float32)
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    centered = bv - bv.mean(axis=1, keepdims=True)
    out["mag_slope"] = centered.dot(xc) / denom
    out["mag_curvature"] = (out["u"] - 2 * out["r"] + out["z"]).astype("float32")
    out["blue_curvature"] = (out["u"] - 2 * out["g"] + out["r"]).astype("float32")
    out["red_curvature"] = (out["r"] - 2 * out["i"] + out["z"]).astype("float32")

    out["redshift_abs"] = out["redshift"].abs().astype("float32")
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"].to_numpy(np.float32)).astype("float32")
    out["redshift_is_neg"] = (out["redshift"] < 0).astype("int8")
    out["spectral_type_calc"] = spectral_type(out["g"], out["r"])
    out["galaxy_population_calc"] = galaxy_population(out["u"], out["r"])
    out["spectral_type"] = cat_key(out["spectral_type"])
    out["galaxy_population"] = cat_key(out["galaxy_population"])
    out["spectral_x_pop"] = cat_key(out["spectral_type"]) + "__" + cat_key(out["galaxy_population"])
    out["spectral_calc_x_pop_calc"] = (cat_key(out["spectral_type_calc"]) + "__"
                                       + cat_key(out["galaxy_population_calc"]))
    return out.replace([np.inf, -np.inf], np.nan)


def add_pairwise_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"] if c in out.columns]
    for c in cols:
        out[f"{c}_x_redshift"] = (out[c] * out["redshift"]).astype("float32")
        out[f"{c}_abs"] = out[c].abs().astype("float32")
    if {"u_g", "g_r"}.issubset(out.columns):
        ug, gr = out["u_g"].to_numpy(np.float32), out["g_r"].to_numpy(np.float32)
        out["color_plane_radius_ug_gr"] = np.sqrt(ug ** 2 + gr ** 2)
        out["color_plane_angle_ug_gr"] = np.arctan2(ug, gr + EPS)
    if {"r_i", "i_z"}.issubset(out.columns):
        ri, iz = out["r_i"].to_numpy(np.float32), out["i_z"].to_numpy(np.float32)
        out["color_plane_radius_ri_iz"] = np.sqrt(ri ** 2 + iz ** 2)
        out["color_plane_angle_ri_iz"] = np.arctan2(ri, iz + EPS)
    return out


def add_quantile_bin_features(df: pd.DataFrame, mask: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """cdeotte's qbin recipe over the shared helper: fixed cols, q=16/64/256, 3 crosses."""
    cols = RAW_NUM_COLS + [c for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean", "mag_range"]
                           if c in df.columns]
    return _add_quantile_bin_features(
        df, mask, cols=cols, quantiles=(16, 64, 256),
        crosses=[("alpha_qbin64", "delta_qbin64"), ("u_g_qbin64", "g_r_qbin64"),
                 ("redshift_qbin64", "mag_mean_qbin64")],
    )


def select_te_cols(df: pd.DataFrame, cat_cols: list[str], max_card: int) -> list[str]:
    # TE_SOURCE == 'all' in the notebook → keep every categorical under the cap.
    return select_low_cardinality_cols(df, cat_cols, max_card)


def add_original_prior_features(
    df: pd.DataFrame, cols: list[str], orig_mask: np.ndarray, orig_y: np.ndarray
) -> pd.DataFrame:
    """Per-category count + per-class rate measured on ORIGINAL rows (leak-safe: external)."""
    out = df.copy()
    prior_counts = np.bincount(orig_y.astype(np.int32), minlength=len(CLASSES)).astype(np.float64)
    prior = prior_counts / max(prior_counts.sum(), 1.0)
    for c in cols:
        if c not in out.columns:
            continue
        key = cat_key(out[c])
        key_orig = key[orig_mask].reset_index(drop=True)
        tmp = pd.DataFrame({"key": key_orig.to_numpy(), "y": orig_y})
        counts = tmp.groupby("key").size()
        out[f"orig_{c}_count"] = key.map(counts).fillna(0).astype("float32")
        for cls_idx, cls_name in INT_TO_CLASS.items():
            rates = tmp.assign(hit=(tmp["y"] == cls_idx).astype("float32")).groupby("key")["hit"].mean()
            out[f"orig_{c}_prior_{cls_name}"] = key.map(rates).fillna(float(prior[cls_idx])).astype("float32")
    return out


def build_feature_matrix(
    train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, orig_y: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Concatenate train/test/orig, build all target-free features → (X, X_test, cat_cols)."""
    tb = train.drop(columns=[TARGET]).copy(); tb["_source"] = "train"
    eb = test.copy(); eb["_source"] = "test"
    ob = orig.drop(columns=[TARGET]).copy(); ob["_source"] = "orig"
    all_df = pd.concat([tb, eb, ob], axis=0, ignore_index=True)

    all_df = add_public_features(all_df)
    all_df = add_pairwise_geometry_features(all_df)

    src = all_df["_source"].to_numpy()
    tt_mask = np.isin(src, ["train", "test"])
    cat_cols = ["spectral_type", "galaxy_population", "spectral_type_calc",
                "galaxy_population_calc", "spectral_x_pop", "spectral_calc_x_pop_calc"]
    all_df, qbin_cols = add_quantile_bin_features(all_df, tt_mask)
    cat_cols = [c for c in dict.fromkeys(cat_cols + qbin_cols) if c in all_df.columns]

    all_mask = np.isin(src, ["train", "test", "orig"])
    freq_cols = select_te_cols(all_df, cat_cols, TE_MAX_CARDINALITY * 4)
    all_df = add_frequency_features(all_df, freq_cols, all_mask)

    orig_mask = src == "orig"
    prior_cols = select_te_cols(all_df, cat_cols, TE_MAX_CARDINALITY * 2)
    all_df = add_original_prior_features(all_df, prior_cols, orig_mask, orig_y)

    all_df = all_df.drop(columns=[c for c in [ID_COL, "_source"] if c in all_df.columns])
    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    n_tr, n_te = len(tb), len(eb)
    X = all_df.iloc[:n_tr].reset_index(drop=True)
    X_test = all_df.iloc[n_tr:n_tr + n_te].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]
    return X, X_test, cat_cols


# The ~240 features cdeotte's notebook selected (gain-ranked), verbatim. The model uses
# only those present after the fold's TE features are added.
TOP_FEATURES: list[str] = [
    "redshift_u", "u_over_redshift", "z_over_redshift", "g_over_redshift", "g_z",
    "redshift_g", "g_i", "u_i", "u_r_abs", "TE_redshift_qbin64__x__mag_mean_qbin64_QSO",
    "i_over_redshift", "u_r", "g_i_abs", "TE_redshift_qbin16_GALAXY", "redshift_z",
    "TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY", "redshift_abs", "TE_redshift_qbin64_GALAXY",
    "orig_g_qbin64_prior_QSO", "redshift_log1p_abs", "orig_g_qbin16_prior_QSO",
    "orig_redshift_qbin64_prior_GALAXY", "redshift", "TE_redshift_qbin64_QSO", "mag_slope",
    "TE_u_r_qbin64_GALAXY", "TE_alpha_qbin64__x__delta_qbin64_STAR", "r_over_redshift", "flux_g",
    "redshift_i", "flux_std", "g", "TE_alpha_qbin64__x__delta_qbin64_GALAXY",
    "TE_u_g_qbin64__x__g_r_qbin64_STAR", "redshift_is_neg", "g_qbin16", "orig_g_qbin256_prior_QSO",
    "TE_u_r_qbin64_QSO", "flux_range", "mag_std", "orig_g_qbin16_prior_GALAXY", "redshift_r",
    "orig_redshift_qbin64_prior_STAR", "orig_u_r_qbin16_prior_QSO", "u_z",
    "orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO", "TE_redshift_qbin64_STAR",
    "TE_g_qbin64_QSO", "orig_redshift_qbin16_prior_GALAXY", "TE_g_qbin16_QSO", "u_g", "z",
    "orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY", "g_i_x_redshift", "orig_z_qbin16_prior_QSO",
    "orig_alpha_qbin64__x__delta_qbin64_prior_STAR", "color_plane_radius_ug_gr", "i", "flux_z",
    "TE_i_qbin64_QSO", "TE_g_qbin64_GALAXY", "orig_redshift_qbin256_prior_GALAXY", "r",
    "TE_u_r_qbin16_GALAXY", "flux_i", "r_i_x_redshift", "flux_r", "r_z", "orig_i_qbin16_prior_QSO",
    "r_z_x_redshift", "g_r_x_redshift", "orig_mag_range_qbin16_prior_STAR", "r_z_abs", "mag_max",
    "TE_g_qbin16_GALAXY", "orig_mag_range_qbin64_prior_STAR", "TE_i_qbin16_QSO", "flux_min",
    "TE_u_g_qbin64_STAR", "orig_u_qbin16_prior_QSO", "TE_redshift_qbin64__x__mag_mean_qbin64_STAR",
    "orig_u_g_qbin16_prior_STAR", "flux_max", "orig_z_qbin64_prior_QSO", "TE_redshift_qbin16_STAR",
    "mag_range", "TE_u_g_qbin64__x__g_r_qbin64_QSO", "orig_redshift_qbin256_count", "g_r",
    "orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY", "orig_i_qbin256_prior_QSO",
    "redshift_qbin16", "mag_mean_qbin16", "TE_u_g_qbin16_STAR", "TE_z_qbin64_QSO", "u_g_abs",
    "orig_u_r_qbin64_prior_QSO", "mag_min", "orig_r_qbin16_prior_QSO",
    "redshift_qbin64__x__mag_mean_qbin64", "u_r_x_redshift", "orig_i_qbin64_prior_QSO", "u",
    "flux_u", "TE_redshift_qbin16_QSO", "flux_mean", "redshift_qbin64__x__mag_mean_qbin64_freq",
    "TE_alpha_qbin64__x__delta_qbin64_QSO", "redshift_qbin64__x__mag_mean_qbin64_freq_log1p",
    "u_g_x_redshift", "u_qbin16", "TE_g_r_qbin64_GALAXY", "color_plane_radius_ri_iz",
    "orig_z_qbin256_prior_QSO", "redshift_qbin256", "TE_mag_range_qbin64_QSO", "g_r_abs",
    "orig_mag_range_qbin256_prior_STAR", "orig_g_qbin64_prior_GALAXY",
    "orig_mag_range_qbin16_prior_GALAXY", "r_i", "r_qbin16", "TE_r_qbin64_STAR",
    "TE_g_r_qbin64_STAR", "TE_u_r_qbin16_QSO", "orig_u_qbin16_prior_STAR", "z_qbin16_freq_log1p",
    "orig_alpha_qbin256_prior_GALAXY", "alpha_sin", "TE_u_g_qbin64_QSO", "orig_spectral_x_pop_prior_QSO",
    "r_qbin64", "sky_y", "u_g_qbin16", "mag_range_qbin256", "TE_r_i_qbin64_QSO",
    "TE_mag_range_qbin64_GALAXY", "orig_alpha_qbin256_prior_STAR", "alpha_qbin256",
    "orig_u_qbin16_count", "z_qbin16", "delta_cos", "orig_u_qbin64_prior_QSO", "g_qbin64",
    "TE_r_qbin16_QSO", "TE_z_qbin16_QSO", "color_plane_angle_ug_gr", "mag_range_qbin16",
    "u_qbin16_freq_log1p", "orig_delta_qbin256_prior_STAR", "TE_g_qbin64_STAR",
    "orig_delta_qbin256_prior_GALAXY", "z_qbin16_freq", "orig_u_g_qbin64_prior_STAR",
    "orig_mag_range_qbin256_prior_GALAXY", "u_qbin16_freq", "orig_z_qbin16_count",
    "orig_u_r_qbin256_prior_QSO", "orig_g_qbin256_prior_GALAXY", "TE_r_i_qbin16_QSO",
    "orig_i_qbin16_prior_GALAXY", "TE_u_qbin16_QSO", "TE_r_i_qbin16_GALAXY", "blue_curvature",
    "TE_r_i_qbin64_GALAXY", "orig_redshift_qbin64__x__mag_mean_qbin64_prior_STAR", "TE_u_qbin64_STAR",
    "TE_g_r_qbin64_QSO", "TE_u_qbin64_QSO", "mag_mean", "TE_i_qbin16_GALAXY", "TE_u_g_qbin16_QSO",
    "TE_u_g_qbin64_GALAXY", "orig_mag_range_qbin256_prior_QSO", "delta", "delta_sin", "alpha",
    "sky_x", "orig_g_qbin64_prior_STAR", "sky_z", "orig_r_qbin64_prior_QSO", "i_qbin16",
    "redshift_qbin64", "orig_alpha_qbin64__x__delta_qbin64_prior_QSO", "TE_g_r_qbin16_STAR",
    "mag_curvature", "orig_r_qbin256_prior_QSO", "TE_g_qbin16_STAR", "TE_alpha_qbin16_QSO",
    "orig_u_r_qbin16_prior_STAR", "TE_mag_range_qbin16_QSO", "orig_r_i_qbin16_prior_GALAXY",
    "orig_alpha_qbin16_prior_STAR", "orig_delta_qbin16_prior_STAR", "orig_u_qbin256_prior_QSO",
    "orig_g_qbin256_prior_STAR", "orig_r_i_qbin64_prior_QSO", "orig_u_g_qbin16_prior_QSO",
    "orig_mag_range_qbin64_prior_GALAXY", "orig_u_g_qbin256_prior_QSO", "redshift_qbin64_freq",
    "u_r_qbin16", "orig_u_r_qbin256_prior_GALAXY", "orig_u_g_qbin64_prior_QSO",
    "orig_alpha_qbin256_prior_QSO", "orig_u_g_qbin256_prior_STAR", "TE_mag_range_qbin16_STAR",
    "TE_u_g_qbin64__x__g_r_qbin64_GALAXY", "orig_g_r_qbin256_prior_GALAXY", "orig_alpha_qbin256_count",
    "redshift_qbin64_freq_log1p", "orig_delta_qbin256_prior_QSO", "orig_u_qbin256_prior_GALAXY",
    "orig_r_i_qbin256_prior_QSO", "orig_redshift_qbin64__x__mag_mean_qbin64_count",
    "orig_r_i_qbin16_prior_STAR", "u_r_qbin256", "orig_r_qbin256_prior_GALAXY",
    "orig_r_i_qbin256_prior_GALAXY", "orig_mag_mean_qbin256_prior_QSO",
    "orig_u_g_qbin64__x__g_r_qbin64_prior_GALAXY", "orig_u_r_qbin256_prior_STAR",
    "orig_z_qbin256_prior_GALAXY", "orig_i_qbin256_prior_GALAXY",
    "orig_u_g_qbin64__x__g_r_qbin64_prior_STAR", "orig_redshift_qbin256_prior_QSO",
    "orig_u_g_qbin256_prior_GALAXY", "orig_u_qbin256_prior_STAR", "orig_mag_mean_qbin256_prior_GALAXY",
    "orig_spectral_calc_x_pop_calc_count", "orig_r_i_qbin256_prior_STAR", "orig_i_qbin256_prior_STAR",
    "orig_spectral_x_pop_prior_STAR", "orig_r_qbin256_prior_STAR", "orig_delta_qbin256_count",
    "orig_g_r_qbin256_prior_QSO", "orig_z_qbin256_count", "orig_g_qbin256_count",
    "orig_r_qbin256_count", "orig_mag_mean_qbin256_count",
]

# ── Experimental feature engineering ──────────────────────────────────────────
# All 40 names produced by add_experimental_features().
EXPERIMENTAL_FEATURES: list[str] = [
    # v1: physics
    "dist_mod",
    "absmag_u", "absmag_g", "absmag_r", "absmag_i", "absmag_z",
    # v1: redshift transforms
    "redshift_sq", "redshift_sqrt", "inv_1pz",
    # v1: squared colors
    "u_g_sq", "g_r_sq", "r_i_sq", "i_z_sq", "u_z_sq",
    # v1: color cross-products
    "u_g_x_g_r", "g_r_x_r_i", "r_i_x_i_z", "u_g_x_r_i", "g_r_x_i_z",
    # v1: band deviations from per-row mean
    "u_dev", "g_dev", "r_dev", "i_dev", "z_dev",
    # v2: SED shape not in Deotte matrix
    "mid_curvature_gri", "mag_convex_gi", "mag_2ndderiv_std", "sed_residual_norm",
    "phys_spec_idx",
    # v2: aggregate cross-terms
    "redshift_x_mag_std", "ug_x_mag_std", "blue_curv_x_mag_std",
    # v2: scale-free photometric ratios
    "flux_concentration", "mag_var_ratio",
    # v2: categorical interactions
    "rs_x_spec_code", "pop_x_uz",
    # v2: stellar locus geometry
    "locus_dist_ur_gr",
    # v2: rest-frame (de-redshifted) colors
    "u_g_restframe", "g_r_restframe", "r_i_restframe",
]

# Physical spectral index: log-linear slope fitted over SDSS effective wavelengths (nm).
# Unlike mag_slope (equal spacing 0–4), this uses the actual nm grid.
_SDSS_LAMBDA = np.array([354.3, 477.0, 623.1, 762.5, 913.4], dtype=np.float64)
_SDSS_LAMBDA_C = _SDSS_LAMBDA - _SDSS_LAMBDA.mean()
_SDSS_LAMBDA_DENOM = float((_SDSS_LAMBDA_C ** 2).sum())

# Stellar locus in (u−r, g−r) colour space: u_r ≈ 1.4·g_r + 0.15
_LOCUS_SLOPE = 1.4
_LOCUS_INTERCEPT = 0.15
_LOCUS_NORM = float(np.sqrt(1.0 + _LOCUS_SLOPE ** 2))

_SPEC_MAP = {"O/B": 3.0, "A/F": 2.0, "G/K": 1.0, "M": 0.0}


def add_experimental_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add ~40 nonlinear transforms on top of the Deotte feature matrix.

    Assumes df already contains all columns built by build_feature_matrix (color pairs,
    mag aggregates, flux columns, curvatures, spectral_type_calc, galaxy_population_calc,
    etc.).  Input is not mutated; returns (augmented_df, list_of_added_names).
    """
    out = df.copy()
    added: list[str] = []

    rs = out["redshift"].to_numpy(np.float64)
    rs_abs = np.abs(rs)
    bv = out[BANDS].to_numpy(np.float64)
    u_v, g_v, r_v, i_v, z_v = bv[:, 0], bv[:, 1], bv[:, 2], bv[:, 3], bv[:, 4]

    # ── v1: physics & monotone redshift transforms ─────────────────────────
    dm = np.where(rs > 0, 5.0 * np.log10(np.maximum(rs_abs, EPS)), np.nan)
    out["dist_mod"] = dm.astype(np.float32)
    added.append("dist_mod")
    for b in BANDS:
        out[f"absmag_{b}"] = (out[b].to_numpy(np.float64) - dm).astype(np.float32)
        added.append(f"absmag_{b}")

    out["redshift_sq"] = (rs ** 2).astype(np.float32)
    out["redshift_sqrt"] = np.sqrt(rs_abs).astype(np.float32)
    out["inv_1pz"] = (1.0 / (1.0 + rs_abs)).astype(np.float32)
    added += ["redshift_sq", "redshift_sqrt", "inv_1pz"]

    # ── v1: squared colors ─────────────────────────────────────────────────
    for c in ["u_g", "g_r", "r_i", "i_z", "u_z"]:
        if c in out.columns:
            out[f"{c}_sq"] = (pd.to_numeric(out[c], errors="coerce")
                              .to_numpy(np.float32) ** 2)
            added.append(f"{c}_sq")

    # ── v1: color cross-products ───────────────────────────────────────────
    for a, b in [("u_g", "g_r"), ("g_r", "r_i"), ("r_i", "i_z"),
                 ("u_g", "r_i"), ("g_r", "i_z")]:
        if a in out.columns and b in out.columns:
            name = f"{a}_x_{b}"
            out[name] = (pd.to_numeric(out[a], errors="coerce").to_numpy(np.float32) *
                         pd.to_numeric(out[b], errors="coerce").to_numpy(np.float32))
            added.append(name)

    # ── v1: band deviations from per-row mean ─────────────────────────────
    if "mag_mean" in out.columns:
        mean_v = out["mag_mean"].to_numpy(np.float32)
        for b in BANDS:
            out[f"{b}_dev"] = (out[b].to_numpy(np.float32) - mean_v)
            added.append(f"{b}_dev")

    # ── v2: SED curvatures not in Deotte (blue=u-2g+r, mag=u-2r+z, red=r-2i+z) ─
    out["mid_curvature_gri"] = (g_v - 2 * r_v + i_v).astype(np.float32)
    out["mag_convex_gi"] = (g_v - 2 * i_v + z_v).astype(np.float32)
    added += ["mid_curvature_gri", "mag_convex_gi"]

    if "blue_curvature" in out.columns and "red_curvature" in out.columns:
        blue_c = out["blue_curvature"].to_numpy(np.float64)
        red_c = out["red_curvature"].to_numpy(np.float64)
        mid_c = g_v - 2 * r_v + i_v
        out["mag_2ndderiv_std"] = (
            np.stack([blue_c, mid_c, red_c], axis=1).std(axis=1, ddof=0)
        ).astype(np.float32)
        added.append("mag_2ndderiv_std")

    if "mag_slope" in out.columns and "mag_mean" in out.columns:
        ms = out["mag_slope"].to_numpy(np.float64)
        mm = out["mag_mean"].to_numpy(np.float64)
        cx = np.array([-2., -1., 0., 1., 2.])
        predicted = mm[:, None] + ms[:, None] * cx[None, :]
        out["sed_residual_norm"] = (
            np.sqrt(((bv - predicted) ** 2).mean(axis=1))
        ).astype(np.float32)
        added.append("sed_residual_norm")

    log_flux = -0.4 * bv
    out["phys_spec_idx"] = (
        (log_flux * _SDSS_LAMBDA_C[None, :]).sum(axis=1) / _SDSS_LAMBDA_DENOM
    ).astype(np.float32)
    added.append("phys_spec_idx")

    # ── v2: aggregate cross-terms ──────────────────────────────────────────
    if "mag_std" in out.columns:
        mag_std = out["mag_std"].to_numpy(np.float32)
        out["redshift_x_mag_std"] = (rs_abs * mag_std).astype(np.float32)
        added.append("redshift_x_mag_std")
        if "u_g" in out.columns:
            out["ug_x_mag_std"] = (
                pd.to_numeric(out["u_g"], errors="coerce").to_numpy(np.float32) * mag_std
            )
            added.append("ug_x_mag_std")
        if "blue_curvature" in out.columns:
            out["blue_curv_x_mag_std"] = (
                out["blue_curvature"].to_numpy(np.float32) * mag_std
            )
            added.append("blue_curv_x_mag_std")

    # ── v2: scale-free photometric ratios ─────────────────────────────────
    if "flux_std" in out.columns and "flux_mean" in out.columns:
        out["flux_concentration"] = (
            out["flux_std"].to_numpy(np.float32) /
            out["flux_mean"].to_numpy(np.float32).clip(EPS, None)
        ).astype(np.float32)
        added.append("flux_concentration")
    if "mag_std" in out.columns and "mag_mean" in out.columns:
        out["mag_var_ratio"] = (
            out["mag_std"].to_numpy(np.float32) /
            (np.abs(out["mag_mean"].to_numpy(np.float32)) + EPS)
        ).astype(np.float32)
        added.append("mag_var_ratio")

    # ── v2: categorical interactions ───────────────────────────────────────
    spec_col = "spectral_type_calc" if "spectral_type_calc" in out.columns else "spectral_type"
    if spec_col in out.columns:
        spec_code = (
            out[spec_col].astype(str).map(_SPEC_MAP).fillna(-1.0).to_numpy(np.float64)
        )
        out["rs_x_spec_code"] = (rs * spec_code).astype(np.float32)
        added.append("rs_x_spec_code")

    pop_col = "galaxy_population_calc" if "galaxy_population_calc" in out.columns else "galaxy_population"
    if pop_col in out.columns and "u_z" in out.columns:
        pop_code = (
            out[pop_col].astype(str)
            .map({"Red_Sequence": 1.0, "Blue_Cloud": 0.0}).fillna(0.5)
            .to_numpy(np.float64)
        )
        out["pop_x_uz"] = (
            pop_code * pd.to_numeric(out["u_z"], errors="coerce").to_numpy(np.float64)
        ).astype(np.float32)
        added.append("pop_x_uz")

    # ── v2: stellar locus geometry ─────────────────────────────────────────
    if "u_r" in out.columns and "g_r" in out.columns:
        ur = pd.to_numeric(out["u_r"], errors="coerce").to_numpy(np.float64)
        gr = pd.to_numeric(out["g_r"], errors="coerce").to_numpy(np.float64)
        out["locus_dist_ur_gr"] = (
            np.abs(ur - _LOCUS_SLOPE * gr - _LOCUS_INTERCEPT) / _LOCUS_NORM
        ).astype(np.float32)
        added.append("locus_dist_ur_gr")

    # ── v2: rest-frame (de-redshifted) colours: colour / (1+|z|) ──────────
    inv1pz = 1.0 / (1.0 + rs_abs)
    for c in ["u_g", "g_r", "r_i"]:
        if c in out.columns:
            out[f"{c}_restframe"] = (
                pd.to_numeric(out[c], errors="coerce").to_numpy(np.float64) * inv1pz
            ).astype(np.float32)
            added.append(f"{c}_restframe")

    return out.replace([np.inf, -np.inf], np.nan), added
