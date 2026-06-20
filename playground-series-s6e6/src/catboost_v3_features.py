"""Faithful pandas/numpy port of cdeotte's CAT_v3 GPU feature builder (CV 0.96897).

Original is RAPIDS (cuDF/cupy). This reproduces build_features_gpu exactly on CPU:
a large mix of numeric features + HIGH-CARDINALITY CATEGORICAL features (qbins, round/floor/
mod cats, hashed PAIR/TRIO combos) that CatBoost consumes via ordered target statistics
(max_ctr_complexity=3) — the lever our prior CatBoost ports missed. The original SDSS17 rows
are appended (with recomputed spectral_type/galaxy_population) so build sees train+test+original.

Returns (X_train, X_test, X_original, cat_cols) as pandas; cat cols are int32 codes, numeric float32.
Leak-safe: every feature is a deterministic function of a row's own columns (qbin edges fit on the
full concatenated feature distribution — unsupervised, applied identically to all splits).
"""

from itertools import combinations

import numpy as np
import pandas as pd

RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]
CLASSES = ["GALAXY", "QSO", "STAR"]
EPS = np.float32(1e-6)
INT_MIN, INT_MAX = -2147483648, 2147483647


def _finite(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return _finite(a / (b + EPS))


def _cat_from(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=float(INT_MIN), posinf=float(INT_MAX), neginf=float(INT_MIN))
    return arr.astype(np.int64).astype(np.int32)


def _qbin(arr, bins):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if valid.sum() <= 1:
        return np.full(arr.shape, -1, dtype=np.int32)
    qs = np.linspace(0, 1, bins + 1, dtype=np.float64)[1:-1]
    edges = np.quantile(arr[valid].astype(np.float64), qs)
    edges = np.unique(edges[np.isfinite(edges)])
    if len(edges) == 0:
        codes = np.zeros(arr.shape, dtype=np.int32)
    else:
        codes = np.searchsorted(edges, arr, side="right").astype(np.int32)
    return np.where(valid, codes, np.int32(-1)).astype(np.int32)


def _floor_cat(arr):
    return _cat_from(np.floor(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=-999999.0)))


def _round_cat(arr, decimals):
    scale = np.float32(10 ** decimals)
    return _cat_from(np.rint(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=-999999.0) * scale))


def _hash2(aa, bb):
    aa = np.asarray(aa, dtype=np.int64)
    bb = np.asarray(bb, dtype=np.int64)
    return _cat_from(((aa + 1_000_003) * 1_000_003 + (bb + 9_176)) % 2_147_483_647)


def _hash3(aa, bb, cc):
    aa = np.asarray(aa, dtype=np.int64)
    bb = np.asarray(bb, dtype=np.int64)
    cc = np.asarray(cc, dtype=np.int64)
    v = (((aa + 1_000_003) * 1_000_003 + (bb + 9_176)) * 1_000_003 + (cc + 17_191)) % 2_147_483_647
    return _cat_from(v)


def _int_col(df, name):
    """Int64 view of a cat column with NA -> -1 (matches cudf to_cupy na_value=-1)."""
    return df[name].fillna(-1).astype(np.int64).to_numpy()


def spectral_type_from_gr(g, r):
    rg = np.asarray(r, dtype=np.float32) - np.asarray(g, dtype=np.float32)
    labels = np.array(["M", "G/K", "A/F", "O/B"])
    idx = np.digitize(rg, [-1.0, -0.5, 0.0])  # 0..3
    return labels[idx]


def galaxy_population_from_ur(u, r):
    ur = np.asarray(u, dtype=np.float32) - np.asarray(r, dtype=np.float32)
    labels = np.array(["Blue_Cloud", "Red_Sequence"])
    return labels[(ur > 2.2).astype(int)]


def build_features_cat_v3(train_df, test_df, original_df):
    """train_df/test_df: competition frames (pandas, raw cols + cats as str, train has 'class').
    original_df: SDSS17 raw frame (pandas) with photometry + 'class'. Returns X, X_test, X_orig, cat_cols.
    """
    spec_map = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
    pop_map = {"Blue_Cloud": 0, "Red_Sequence": 1}

    def prep_comp(df):
        out = pd.DataFrame()
        for c in RAW_NUM_COLS:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        out["spectral_type"] = df["spectral_type"].astype(str)
        out["galaxy_population"] = df["galaxy_population"].astype(str)
        return out

    def prep_orig(df):
        out = pd.DataFrame()
        for c in RAW_NUM_COLS:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        out["spectral_type"] = spectral_type_from_gr(out["g"], out["r"])
        out["galaxy_population"] = galaxy_population_from_ur(out["u"], out["r"])
        return out

    tr, te, orig = prep_comp(train_df), prep_comp(test_df), prep_orig(original_df)
    n_train, n_test = len(tr), len(te)
    gdf = pd.concat([tr, te, orig], axis=0, ignore_index=True)

    for c in RAW_NUM_COLS:
        gdf[c] = gdf[c].astype(np.float32)
    gdf["spectral_type"] = gdf["spectral_type"].map(spec_map).fillna(-1).astype(np.int32)
    gdf["galaxy_population"] = gdf["galaxy_population"].map(pop_map).fillna(-1).astype(np.int32)
    cat_cols = ["spectral_type", "galaxy_population"]

    # pairwise color differences + abs
    for a, b in combinations(BANDS, 2):
        gdf[f"{a}_{b}"] = (gdf[a] - gdf[b]).astype(np.float32)
        gdf[f"{a}_{b}_abs"] = gdf[f"{a}_{b}"].abs().astype(np.float32)

    mags = np.stack([gdf[b].to_numpy(np.float32) for b in BANDS], axis=1)
    gdf["mag_mean"] = np.nanmean(mags, axis=1).astype(np.float32)
    gdf["mag_std"] = np.nanstd(mags, axis=1).astype(np.float32)
    gdf["mag_min"] = np.nanmin(mags, axis=1).astype(np.float32)
    gdf["mag_max"] = np.nanmax(mags, axis=1).astype(np.float32)
    gdf["mag_range"] = (gdf["mag_max"] - gdf["mag_min"]).astype(np.float32)
    gdf["mag_argmin"] = _cat_from(np.nanargmin(mags, axis=1))
    gdf["mag_argmax"] = _cat_from(np.nanargmax(mags, axis=1))
    cat_cols += ["mag_argmin", "mag_argmax"]

    x_center = np.arange(len(BANDS), dtype=np.float32) - np.arange(len(BANDS), dtype=np.float32).mean()
    centered = mags - np.nanmean(mags, axis=1)[:, None]
    gdf["mag_slope"] = (centered.dot(x_center) / np.sum(x_center ** 2)).astype(np.float32)
    gdf["mag_curvature"] = (gdf["u"] - 2 * gdf["r"] + gdf["z"]).astype(np.float32)
    gdf["blue_curvature"] = (gdf["u"] - 2 * gdf["g"] + gdf["r"]).astype(np.float32)
    gdf["red_curvature"] = (gdf["r"] - 2 * gdf["i"] + gdf["z"]).astype(np.float32)

    redshift = gdf["redshift"].to_numpy(np.float32)
    gdf["redshift_abs"] = gdf["redshift"].abs().astype(np.float32)
    gdf["redshift_log1p_abs"] = np.log1p(np.abs(redshift)).astype(np.float32)
    gdf["redshift_sq"] = (gdf["redshift"] ** 2).astype(np.float32)
    gdf["redshift_cbrt"] = np.cbrt(redshift).astype(np.float32)
    gdf["redshift_is_neg"] = _cat_from(redshift < 0)
    gdf["redshift_lt_002"] = _cat_from(redshift < 0.02)
    gdf["redshift_gt_07"] = _cat_from(redshift > 0.7)
    cat_cols += ["redshift_is_neg", "redshift_lt_002", "redshift_gt_07"]

    gdf["g_over_redshift"] = _safe_div(gdf["g"].to_numpy(), np.abs(redshift))
    gdf["i_over_redshift"] = _safe_div(gdf["i"].to_numpy(), np.abs(redshift))
    gdf["z_over_redshift"] = _safe_div(gdf["z"].to_numpy(), np.abs(redshift))
    gdf["z_over_g"] = _safe_div(gdf["z"].to_numpy(), gdf["g"].to_numpy())
    gdf["z2_over_g2"] = _safe_div(gdf["z"].to_numpy() ** 2, gdf["g"].to_numpy() ** 2)
    gdf["log_z_over_log_g"] = _safe_div(np.log1p(np.abs(gdf["z"].to_numpy())), np.log1p(np.abs(gdf["g"].to_numpy())))
    gdf["sqrt_z_over_sqrt_g"] = _safe_div(np.sqrt(np.abs(gdf["z"].to_numpy())), np.sqrt(np.abs(gdf["g"].to_numpy())))
    for b in BANDS:
        gdf[f"redshift_x_{b}"] = (gdf["redshift"] * gdf[b]).astype(np.float32)

    gdf["ug_gr_ratio"] = _safe_div(gdf["u_g"].to_numpy(), gdf["g_r"].to_numpy())
    gdf["gr_ri_ratio"] = _safe_div(gdf["g_r"].to_numpy(), gdf["r_i"].to_numpy())
    gdf["ri_iz_ratio"] = _safe_div(gdf["r_i"].to_numpy(), gdf["i_z"].to_numpy())
    color_mat = np.stack([gdf[c].to_numpy(np.float32) for c in ["u_g", "g_r", "r_i", "i_z"]], axis=1)
    gdf["color_mean"] = np.nanmean(color_mat, axis=1).astype(np.float32)
    gdf["color_std"] = np.nanstd(color_mat, axis=1).astype(np.float32)
    gdf["color_abs_sum"] = np.nansum(np.abs(color_mat), axis=1).astype(np.float32)

    gr = np.clip(gdf["g_r"].to_numpy(np.float32), -5, 5)
    ur = np.clip(gdf["u_r"].to_numpy(np.float32), -5, 8)
    gdf["color_temp_gr_proxy"] = _finite(4600.0 * ((1.0 / (0.92 * gr + 1.7)) + (1.0 / (0.92 * gr + 0.62))))
    gdf["uv_excess_proxy"] = (gdf["u_g"] - (0.75 * gdf["g_r"] + 0.18)).astype(np.float32)
    gdf["red_sequence_score_proxy"] = (ur - 2.2).astype(np.float32)

    alpha_rad = gdf["alpha"].to_numpy(np.float32) * np.float32(np.pi / 180.0)
    delta_rad = gdf["delta"].to_numpy(np.float32) * np.float32(np.pi / 180.0)
    gdf["alpha_sin"] = np.sin(alpha_rad).astype(np.float32)
    gdf["alpha_cos"] = np.cos(alpha_rad).astype(np.float32)
    gdf["delta_sin"] = np.sin(delta_rad).astype(np.float32)
    gdf["delta_cos"] = np.cos(delta_rad).astype(np.float32)
    gdf["sky_x"] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype(np.float32)
    gdf["sky_y"] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype(np.float32)
    gdf["sky_z"] = np.sin(delta_rad).astype(np.float32)

    fluxes = []
    for b in BANDS:
        clipped = np.clip(gdf[b].to_numpy(np.float32), -30, 30)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype(np.float32)
        gdf[f"flux_{b}"] = flux
        gdf[f"log_flux_{b}"] = np.log1p(flux).astype(np.float32)
        fluxes.append(flux)
    fv = np.stack(fluxes, axis=1)
    gdf["flux_mean"] = np.nanmean(fv, axis=1).astype(np.float32)
    gdf["flux_std"] = np.nanstd(fv, axis=1).astype(np.float32)
    gdf["flux_range"] = (np.nanmax(fv, axis=1) - np.nanmin(fv, axis=1)).astype(np.float32)
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("r", "z")]:
        color = np.clip(gdf[a].to_numpy(np.float32) - gdf[b].to_numpy(np.float32), -20, 20)
        gdf[f"flux_ratio_{a}_{b}"] = np.exp(np.float32(-0.921034) * color).astype(np.float32)

    rg2 = gdf["r"].to_numpy(np.float32) - gdf["g"].to_numpy(np.float32)
    spec_calc = np.where(rg2 <= -1.0, 0, np.where(rg2 <= -0.5, 1, np.where(rg2 <= 0.0, 2, 3)))
    pop_calc = np.where((gdf["u"].to_numpy(np.float32) - gdf["r"].to_numpy(np.float32)) <= 2.2, 0, 1)
    gdf["spectral_type_calc"] = _cat_from(spec_calc)
    gdf["galaxy_population_calc"] = _cat_from(pop_calc)
    gdf["spectral_x_pop"] = _cat_from(_int_col(gdf, "spectral_type") * 10 + _int_col(gdf, "galaxy_population"))
    gdf["spectral_calc_x_pop_calc"] = _cat_from(spec_calc * 10 + pop_calc)
    cat_cols += ["spectral_type_calc", "galaxy_population_calc", "spectral_x_pop", "spectral_calc_x_pop_calc"]

    for c in RAW_NUM_COLS:
        gdf[f"{c}_floor_cat"] = _floor_cat(gdf[c].to_numpy())
        cat_cols.append(f"{c}_floor_cat")

    q_specs = {c: [32, 100, 500] for c in RAW_NUM_COLS}
    for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "r_z", "mag_range", "mag_std"]:
        q_specs.setdefault(c, [64])
    for c, bins_list in q_specs.items():
        for bins in bins_list:
            gdf[f"{c}_q{bins}_cat"] = _qbin(gdf[c].to_numpy(), bins)
            cat_cols.append(f"{c}_q{bins}_cat")

    round_specs = {
        "alpha": 1, "delta": 1, "u": 2, "g": 2, "r": 2, "i": 2, "z": 2,
        "redshift": 4, "u_g": 3, "g_r": 3, "r_i": 3, "i_z": 3,
        "u_r": 3, "r_z": 3, "mag_range": 3,
    }
    for c, dec in round_specs.items():
        gdf[f"{c}_round{dec}_cat"] = _round_cat(gdf[c].to_numpy(), dec)
        cat_cols.append(f"{c}_round{dec}_cat")

    for c in RAW_NUM_COLS:
        vals = np.abs(gdf[c].to_numpy(np.float32))
        ints = np.floor(np.nan_to_num(vals, nan=0.0)).astype(np.int64)
        frac = np.floor((vals - np.floor(vals)) * 20).astype(np.int32)
        dec = np.floor((vals - np.floor(vals)) * 1000).astype(np.int32)
        for suffix, arr in [("mod10", ints % 10), ("mod100", ints % 100), ("frac20", frac), ("decimal1000", dec)]:
            gdf[f"{c}_{suffix}_cat"] = _cat_from(arr)
            cat_cols.append(f"{c}_{suffix}_cat")

    manual_combos = [
        ("alpha_floor_cat", "delta_floor_cat"), ("u_floor_cat", "z_floor_cat"),
        ("spectral_type", "galaxy_population"), ("spectral_x_pop", "redshift_q64_cat"),
        ("u_r_q64_cat", "redshift_q64_cat"), ("g_r_q64_cat", "mag_range_q64_cat"),
        ("alpha_q100_cat", "delta_q100_cat"), ("u_q100_cat", "z_q100_cat"),
        ("spectral_x_pop", "redshift_q100_cat"),
    ]
    for a, b in manual_combos:
        if a in gdf.columns and b in gdf.columns:
            gdf[f"COMBO_{a}__{b}"] = _hash2(_int_col(gdf, a), _int_col(gdf, b))
            cat_cols.append(f"COMBO_{a}__{b}")

    combo_bases = [
        "spectral_type", "galaxy_population", "spectral_x_pop",
        "alpha_floor_cat", "delta_floor_cat", "u_floor_cat", "z_floor_cat",
        "alpha_q100_cat", "delta_q100_cat", "u_q100_cat", "z_q100_cat",
        "redshift_q64_cat", "u_r_q64_cat", "g_r_q64_cat", "mag_range_q64_cat",
    ]
    combo_bases = [c for c in combo_bases if c in gdf.columns]
    bases = combo_bases[:10]
    for a, b in combinations(bases, 2):
        gdf[f"PAIR_{a}__{b}"] = _hash2(_int_col(gdf, a), _int_col(gdf, b))
        cat_cols.append(f"PAIR_{a}__{b}")
    for trio in [bases[:3], bases[3:6], bases[6:9]]:
        if len(trio) == 3:
            name = "TRIO_" + "__".join(trio)
            gdf[name] = _hash3(_int_col(gdf, trio[0]), _int_col(gdf, trio[1]), _int_col(gdf, trio[2]))
            cat_cols.append(name)

    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in gdf.columns]
    for c in cat_cols:
        gdf[c] = gdf[c].fillna(-1).astype(np.int32)
    for c in gdf.columns:
        if c not in cat_cols:
            gdf[c] = gdf[c].astype(np.float32)

    X = gdf.iloc[:n_train].reset_index(drop=True)
    X_test = gdf.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_original = gdf.iloc[n_train + n_test:].reset_index(drop=True)
    return X, X_test, X_original, cat_cols
