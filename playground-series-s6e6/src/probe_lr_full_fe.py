"""LR Feature Importance Probe — Full Deotte Feature Matrix + Extra Nonlinear Transforms (S6E6)

Goal: find which features (including physics-motivated transforms NOT in the Deotte matrix)
carry unique signal when a linear model is forced to be explicit about them.

Two models:
  L2 LR (C=1.0)  → preserves ALL correlated signals; coef magnitude = raw importance
  L1 LR (C=0.1)  → sparse selection; surviving features are UNIQUELY informative

The L1/L2 gap flags features "swallowed by correlates" in the 240-feature GBDT matrix — these
are the candidates that would benefit from being the only representative of their cluster.

Extra transforms added on top of the Deotte matrix:
  dist_mod = 5·log10(z)         physics: luminosity distance (NaN for z≤0 = STAR signal)
  absmag_{u,g,r,i,z} = m − μ   physics: absolute magnitude proxy
  redshift_sq                   quadratic z (QSO large-z curvature)
  redshift_sqrt                 sqrt(|z|) (compresses high-z end for linear model)
  inv_1pz = 1/(1+|z|)           cosmological scale factor a
  {u_g,g_r,r_i,i_z,u_z}_sq     squared colors (LR can't compute x² from x)
  {ug×gr, gr×ri, ri×iz} cross   color cross-products (LR can't compute xy from x,y)
  {u,g,r,i,z}_dev = b − mag_mean band deviations from centroid

Uses a stratified 100k-row subsample for runtime (3-fold). Diagnostic only, not a stacking base.

Run:  python src/probe_lr_full_fe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from deotte_features import (
    BANDS, CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET,
    TE_INNER_SPLITS, TE_SMOOTH, TOP_FEATURES,
    build_feature_matrix, cat_key,
)


def te_sources(top: list[str], cat_cols: list[str]) -> list[str]:
    return [c for c in cat_cols if any(f.startswith(f"TE_{c}_") for f in top)]

from data_science_stuff.kaggle.encoding import add_fold_safe_target_encoding
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED = 42
N_SPLITS = 3      # use 5 for more reliable estimates (slower)
SUBSAMPLE = 100_000
C_L2 = 1.0
C_L1 = 0.1
EPS = 1e-6

# Feature group tags for output
_EXTRA_PREFIX = {
    # v1 batch
    "dist_mod", "absmag_u", "absmag_g", "absmag_r", "absmag_i", "absmag_z",
    "redshift_sq", "redshift_sqrt", "inv_1pz",
    "u_g_sq", "g_r_sq", "r_i_sq", "i_z_sq", "u_z_sq",
    "u_g_x_g_r", "g_r_x_r_i", "r_i_x_i_z", "u_g_x_r_i", "g_r_x_i_z",
    "u_dev", "g_dev", "r_dev", "i_dev", "z_dev",
    # v2 batch (brainstorm)
    "mid_curvature_gri", "mag_convex_gi", "mag_2ndderiv_std", "sed_residual_norm",
    "phys_spec_idx",
    "ug_x_ri", "gr_x_iz",
    "redshift_x_mag_std", "ug_x_mag_std", "blue_curv_x_mag_std",
    "flux_concentration", "mag_var_ratio",
    "rs_x_spec_code", "pop_x_uz",
    "locus_dist_ur_gr",
    "ug_restframe", "gr_restframe", "ri_restframe",
}


def _group(name: str) -> str:
    if name in _EXTRA_PREFIX:
        return "extra"
    if name.startswith("TE_"):
        return "te"
    if name.startswith("orig_"):
        return "prior"
    if name.endswith("_freq") or name.endswith("_freq_log1p"):
        return "freq"
    if name.startswith("flux_"):
        return "flux"
    if name in {"mag_mean", "mag_std", "mag_min", "mag_max", "mag_range",
                "mag_argmin", "mag_argmax", "mag_slope", "mag_curvature",
                "blue_curvature", "red_curvature"}:
        return "mag_agg"
    if name in {"alpha_sin", "alpha_cos", "delta_sin", "delta_cos",
                "sky_x", "sky_y", "sky_z"}:
        return "sky"
    if name in {"redshift_abs", "redshift_log1p_abs", "redshift_is_neg",
                "redshift_sq", "redshift_sqrt", "inv_1pz"} or name.startswith("redshift_"):
        return "redshift_x"
    if name.endswith("_over_redshift") or name.endswith("_x_redshift"):
        return "redshift_x"
    if name.startswith("color_plane_"):
        return "geo"
    if "_abs" in name and name.split("_abs")[0] in {
            "u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z", "u_z"}:
        return "colors"
    if name in {"u_g", "g_r", "r_i", "i_z", "u_r", "u_i", "u_z",
                "g_i", "g_z", "r_z"}:
        return "colors"
    if name in BANDS or name in {"alpha", "delta", "redshift"}:
        return "raw"
    return "other"


# ── Extra nonlinear transforms ─────────────────────────────────────────────

def add_extra_transforms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []

    rs = out["redshift"].to_numpy(np.float64)

    # Physics: distance modulus (NaN for non-positive redshift — itself a STAR signal)
    dm = np.where(rs > 0, 5.0 * np.log10(np.maximum(rs, EPS)), np.nan)
    out["dist_mod"] = dm.astype(np.float32)
    added.append("dist_mod")

    for b in BANDS:
        bv = out[b].to_numpy(np.float64)
        out[f"absmag_{b}"] = (bv - dm).astype(np.float32)
        added.append(f"absmag_{b}")

    out["redshift_sq"] = (rs ** 2).astype(np.float32)
    out["redshift_sqrt"] = np.sqrt(np.abs(rs)).astype(np.float32)
    out["inv_1pz"] = (1.0 / (1.0 + np.abs(rs))).astype(np.float32)
    added += ["redshift_sq", "redshift_sqrt", "inv_1pz"]

    for c in ["u_g", "g_r", "r_i", "i_z", "u_z"]:
        if c in out.columns:
            out[f"{c}_sq"] = (out[c].to_numpy(np.float32) ** 2)
            added.append(f"{c}_sq")

    for a, b in [("u_g", "g_r"), ("g_r", "r_i"), ("r_i", "i_z"),
                 ("u_g", "r_i"), ("g_r", "i_z")]:
        if a in out.columns and b in out.columns:
            name = f"{a}_x_{b}"
            out[name] = (out[a].to_numpy(np.float32) * out[b].to_numpy(np.float32))
            added.append(name)

    if "mag_mean" in out.columns:
        mean_v = out["mag_mean"].to_numpy(np.float32)
        for b in BANDS:
            out[f"{b}_dev"] = (out[b].to_numpy(np.float32) - mean_v)
            added.append(f"{b}_dev")

    return out.replace([np.inf, -np.inf], np.nan), added


# SDSS effective wavelengths (nm) for the physical spectral index
_SDSS_LAMBDA = np.array([354.3, 477.0, 623.1, 762.5, 913.4], dtype=np.float64)
_SDSS_LAMBDA_C = _SDSS_LAMBDA - _SDSS_LAMBDA.mean()
_SDSS_LAMBDA_DENOM = float((_SDSS_LAMBDA_C ** 2).sum())

# Stellar locus in (u-r, g-r) colour space: u_r ≈ 1.4 * g_r + 0.15
_LOCUS_SLOPE = 1.4
_LOCUS_INTERCEPT = 0.15
_LOCUS_NORM = float(np.sqrt(1.0 + _LOCUS_SLOPE ** 2))


def add_v2_transforms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Brainstorm batch: SED shape features, disjoint colour products, aggregate cross-terms,
    scale-free ratios, categorical interactions, stellar-locus geometry, rest-frame colours."""
    out = df.copy()
    added: list[str] = []
    rs_abs = out["redshift"].to_numpy(np.float64).__abs__()

    bv = out[BANDS].to_numpy(np.float64)      # (n, 5)
    u_v, g_v, r_v, i_v, z_v = bv[:, 0], bv[:, 1], bv[:, 2], bv[:, 3], bv[:, 4]

    # ── SED shape ──────────────────────────────────────────────────────────
    # Two curvatures not in the Deotte matrix
    # Deotte has: blue=u-2g+r, mag=u-2r+z, red=r-2i+z
    out["mid_curvature_gri"] = (g_v - 2*r_v + i_v).astype(np.float32)
    out["mag_convex_gi"] = (g_v - 2*i_v + z_v).astype(np.float32)
    added += ["mid_curvature_gri", "mag_convex_gi"]

    # Std of the 3 consecutive 2nd derivatives per row (SED smoothness / jaggedness)
    if "blue_curvature" in out.columns and "red_curvature" in out.columns:
        blue_c = out["blue_curvature"].to_numpy(np.float64)
        red_c = out["red_curvature"].to_numpy(np.float64)
        mid_c = g_v - 2*r_v + i_v
        derivs = np.stack([blue_c, mid_c, red_c], axis=1)
        out["mag_2ndderiv_std"] = derivs.std(axis=1, ddof=0).astype(np.float32)
        added.append("mag_2ndderiv_std")

    # RMS residual after removing linear SED slope (≠ mag_std which includes the slope)
    if "mag_slope" in out.columns and "mag_mean" in out.columns:
        ms = out["mag_slope"].to_numpy(np.float64)
        mm = out["mag_mean"].to_numpy(np.float64)
        cx = np.array([-2., -1., 0., 1., 2.])
        predicted = mm[:, None] + ms[:, None] * cx[None, :]
        out["sed_residual_norm"] = np.sqrt(((bv - predicted) ** 2).mean(axis=1)).astype(np.float32)
        added.append("sed_residual_norm")

    # Physical spectral index: log-linear slope over ACTUAL nm wavelengths
    # (mag_slope uses equal spacing 0-4; this uses nm grid [354,477,623,763,913])
    log_flux = -0.4 * bv
    out["phys_spec_idx"] = (
        (log_flux * _SDSS_LAMBDA_C[None, :]).sum(axis=1) / _SDSS_LAMBDA_DENOM
    ).astype(np.float32)
    added.append("phys_spec_idx")

    # ── Disjoint colour cross-products ─────────────────────────────────────
    # Existing: g_r×r_i, r_i×i_z (adjacent pairs)
    # New: u_g×r_i (bands u,g cross with r,i — no band overlap)
    #      g_r×i_z (bands g,r cross with i,z — no band overlap)
    if "u_g" in out.columns and "r_i" in out.columns:
        out["ug_x_ri"] = (out["u_g"].to_numpy(np.float32) *
                          out["r_i"].to_numpy(np.float32))
        added.append("ug_x_ri")
    if "g_r" in out.columns and "i_z" in out.columns:
        out["gr_x_iz"] = (out["g_r"].to_numpy(np.float32) *
                          out["i_z"].to_numpy(np.float32))
        added.append("gr_x_iz")

    # ── Aggregate cross-terms ───────────────────────────────────────────────
    if "mag_std" in out.columns:
        mag_std = out["mag_std"].to_numpy(np.float32)
        out["redshift_x_mag_std"] = (rs_abs * mag_std).astype(np.float32)
        added.append("redshift_x_mag_std")
        if "u_g" in out.columns:
            out["ug_x_mag_std"] = (out["u_g"].to_numpy(np.float32) * mag_std)
            added.append("ug_x_mag_std")
        if "blue_curvature" in out.columns:
            bc = out["blue_curvature"].to_numpy(np.float32)
            out["blue_curv_x_mag_std"] = (bc * mag_std)
            added.append("blue_curv_x_mag_std")

    # ── Scale-free photometric ratios ──────────────────────────────────────
    if "flux_std" in out.columns and "flux_mean" in out.columns:
        out["flux_concentration"] = (
            out["flux_std"].to_numpy(np.float32) /
            (out["flux_mean"].to_numpy(np.float32).clip(EPS, None))
        ).astype(np.float32)
        added.append("flux_concentration")
    if "mag_std" in out.columns and "mag_mean" in out.columns:
        out["mag_var_ratio"] = (
            out["mag_std"].to_numpy(np.float32) /
            (np.abs(out["mag_mean"].to_numpy(np.float32)) + EPS)
        ).astype(np.float32)
        added.append("mag_var_ratio")

    # ── Categorical interactions ────────────────────────────────────────────
    # spectral_type → ordinal code (O/B=3, A/F=2, G/K=1, M=0)
    spec_map = {"O/B": 3.0, "A/F": 2.0, "G/K": 1.0, "M": 0.0}
    if "spectral_type" in out.columns:
        spec_code = (
            out["spectral_type"].astype(str)
            .map(spec_map).fillna(-1.0).to_numpy(np.float64)
        )
        out["rs_x_spec_code"] = (out["redshift"].to_numpy(np.float64) * spec_code).astype(np.float32)
        added.append("rs_x_spec_code")

    # galaxy_population → binary (Red_Sequence=1, Blue_Cloud=0) × u_z colour span
    if "galaxy_population" in out.columns and "u_z" in out.columns:
        pop_code = (
            out["galaxy_population"].astype(str)
            .map({"Red_Sequence": 1.0, "Blue_Cloud": 0.0}).fillna(0.5).to_numpy(np.float64)
        )
        out["pop_x_uz"] = (pop_code * out["u_z"].to_numpy(np.float64)).astype(np.float32)
        added.append("pop_x_uz")

    # ── Stellar locus geometry ──────────────────────────────────────────────
    # Stars trace u_r ≈ 1.4*(g_r) + 0.15 in colour space; perpendicular distance
    if "u_r" in out.columns and "g_r" in out.columns:
        ur = out["u_r"].to_numpy(np.float64)
        gr = out["g_r"].to_numpy(np.float64)
        out["locus_dist_ur_gr"] = (
            np.abs(ur - _LOCUS_SLOPE * gr - _LOCUS_INTERCEPT) / _LOCUS_NORM
        ).astype(np.float32)
        added.append("locus_dist_ur_gr")

    # ── Rest-frame (de-redshifted) colours: colour / (1+|z|) ───────────────
    # Not in existing matrix — equivalent to colour × inv_1pz (a genuine cross-term)
    inv1pz = 1.0 / (1.0 + rs_abs)
    for c in ["u_g", "g_r", "r_i"]:
        name = f"{c}_restframe"
        if c in out.columns:
            out[name] = (out[c].to_numpy(np.float64) * inv1pz).astype(np.float32)
            added.append(name)

    return out.replace([np.inf, -np.inf], np.nan), added


# ── Fold-safe target encoding (train/val only — no test needed here) ───────
def add_fold_safe_te(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, te_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return add_fold_safe_target_encoding(
        X_tr, y_tr, [X_va], te_cols, INT_TO_CLASS,
        smooth=TE_SMOOTH, inner_cv=TE_INNER_SPLITS, seed=SEED + 177,
    )

# ── Design matrix helpers ──────────────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in {TARGET, ID_COL}
            and pd.api.types.is_numeric_dtype(df[c])]


def to_matrix(df: pd.DataFrame, cols: list[str],
              medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = df[cols].to_numpy(np.float64)
    if medians is None:
        medians = np.nanmedian(arr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        if mask.any():
            arr[mask, j] = medians[j]
    return arr, medians


# ── Reporting ──────────────────────────────────────────────────────────────

def print_top_features(
    coef: np.ndarray, names: list[str], label: str, n_top: int = 25,
) -> None:
    print(f"\n{'='*65}")
    print(f"  {label} — top {n_top} per class  |coef| (mean across folds)")
    print(f"{'='*65}")
    for cls_idx, cls_name in INT_TO_CLASS.items():
        c = coef[cls_idx]
        order = np.argsort(np.abs(c))[::-1][:n_top]
        print(f"\n  {cls_name}:")
        for rank, fi in enumerate(order):
            g = _group(names[fi])
            marker = " ★" if g == "extra" else ""
            print(f"    {rank+1:2d}. {names[fi]:50s}  {c[fi]:+.4f}  [{g}]{marker}")


def print_extra_section(
    coef_l2: np.ndarray, coef_l1: np.ndarray, names: list[str], extra_cols: list[str],
) -> None:
    print(f"\n{'='*65}")
    print(f"  EXTRA TRANSFORMS — L2 vs L1 coefficients per class")
    print(f"{'='*65}")
    extra_set = set(extra_cols)
    idxs = [i for i, n in enumerate(names) if n in extra_set]
    if not idxs:
        print("  (none found in design matrix)")
        return
    print(f"  {'Feature':40s}  {'L2 GALAXY':>9}  {'L2 QSO':>9}  {'L2 STAR':>9}"
          f"  {'L1 GALAXY':>9}  {'L1 QSO':>9}  {'L1 STAR':>9}")
    for fi in idxs:
        l2 = coef_l2[:, fi]
        l1 = coef_l1[:, fi]
        print(f"  {names[fi]:40s}"
              f"  {l2[0]:+9.4f}  {l2[1]:+9.4f}  {l2[2]:+9.4f}"
              f"  {l1[0]:+9.4f}  {l1[1]:+9.4f}  {l1[2]:+9.4f}")


def print_swallowed(
    coef_l2: np.ndarray, coef_l1: np.ndarray, names: list[str], top_k: int = 20,
) -> None:
    """Features with large L2 |coef| but near-zero L1 |coef| = swallowed by correlates."""
    print(f"\n{'='*65}")
    print(f"  L1/L2 GAP — features with large L2 importance but small L1 importance")
    print(f"  (informative but made redundant by highly-correlated features)")
    print(f"{'='*65}")
    max_l2 = np.abs(coef_l2).max(axis=0)
    max_l1 = np.abs(coef_l1).max(axis=0)
    scale = max(max_l2.max(), 1e-12)
    gap = (max_l2 - max_l1) / scale
    order = np.argsort(gap)[::-1][:top_k]
    for rank, fi in enumerate(order):
        g = _group(names[fi])
        marker = " ★" if g == "extra" else ""
        print(f"  {rank+1:2d}. {names[fi]:50s}  L2={max_l2[fi]:.4f}  L1={max_l1[fi]:.4f}"
              f"  gap={gap[fi]:.4f}  [{g}]{marker}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)

    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_all = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()

    print("Building Deotte feature matrix ...")
    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)

    print("Adding v1 extra transforms ...")
    X, extra_v1 = add_extra_transforms(X)
    print("Adding v2 brainstorm transforms ...")
    X, extra_v2 = add_v2_transforms(X)
    extra_cols = extra_v1 + extra_v2
    print(f"  {len(extra_v1)} v1 + {len(extra_v2)} v2 = {len(extra_cols)} extra transforms")

    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    print(f"Total columns in X: {X.shape[1]}  |  TE sources: {len(TE_COLS)}")

    # Stratified subsample
    rng = np.random.default_rng(SEED)
    if len(X) > SUBSAMPLE:
        idx_parts = []
        for cls_idx in range(len(CLASSES)):
            ci = np.where(y_all == cls_idx)[0]
            n = max(1, int(round(SUBSAMPLE * len(ci) / len(y_all))))
            idx_parts.append(rng.choice(ci, n, replace=False))
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        X_sub = X.iloc[idx].reset_index(drop=True)
        y_sub = y_all[idx]
        print(f"Subsampled {len(X_sub):,} rows from {len(X):,} total "
              f"({','.join(f'{(y_sub==c).sum()}' for c in range(len(CLASSES)))} per class)")
    else:
        X_sub, y_sub = X, y_all

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    feat_names: list[str] = []
    coefs_l2: list[np.ndarray] = []
    coefs_l1: list[np.ndarray] = []
    oof_l2 = np.zeros((len(X_sub), len(CLASSES)), dtype=np.float32)
    oof_l1 = np.zeros((len(X_sub), len(CLASSES)), dtype=np.float32)

    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y_sub)), y_sub), 1):
        print(f"\nFold {fold}/{N_SPLITS} ...")
        X_tr_raw = X_sub.iloc[tri].copy()
        X_va_raw = X_sub.iloc[vai].copy()
        y_tr, y_va = y_sub[tri], y_sub[vai]

        X_tr_raw, X_va_raw = add_fold_safe_te(X_tr_raw, y_tr, X_va_raw, TE_COLS)

        num_cols = get_numeric_cols(X_tr_raw)
        if fold == 1:
            feat_names = num_cols
            print(f"  Design matrix: {len(feat_names)} features")
            extra_set = set(extra_cols)
            n_extra_in_mat = sum(1 for n in feat_names if n in extra_set)
            print(f"  Extra transforms in design matrix: {n_extra_in_mat} "
                  f"(v1={sum(1 for n in feat_names if n in set(extra_v1))}, "
                  f"v2={sum(1 for n in feat_names if n in set(extra_v2))})")

        Xtr, tr_medians = to_matrix(X_tr_raw, num_cols)
        Xva, _ = to_matrix(X_va_raw, num_cols, medians=tr_medians)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)

        # L2
        lr2 = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.0, solver="saga",
            C=C_L2, class_weight="balanced",
            max_iter=200, tol=1e-2, random_state=SEED + fold,
        )
        lr2.fit(Xtr_s, y_tr)
        oof_l2[vai] = lr2.predict_proba(Xva_s)
        coefs_l2.append(lr2.coef_.copy())

        # L1
        lr1 = LogisticRegression(
            penalty="elasticnet", l1_ratio=1.0, solver="saga",
            C=C_L1, class_weight="balanced",
            max_iter=300, tol=1e-2, random_state=SEED + fold,
        )
        lr1.fit(Xtr_s, y_tr)
        oof_l1[vai] = lr1.predict_proba(Xva_s)
        coefs_l1.append(lr1.coef_.copy())

        ba2 = float(balanced_accuracy_score(y_va, np.argmax(oof_l2[vai], axis=1)))
        ba1 = float(balanced_accuracy_score(y_va, np.argmax(oof_l1[vai], axis=1)))
        n_l2_nonzero = int((np.abs(lr2.coef_).max(axis=0) > 1e-6).sum())
        n_l1_nonzero = int((np.abs(lr1.coef_).max(axis=0) > 1e-6).sum())
        print(f"  L2 bal_acc={ba2:.4f} ({n_l2_nonzero} nonzero)  "
              f"L1 bal_acc={ba1:.4f} ({n_l1_nonzero} nonzero/{len(feat_names)})")

    coef_l2 = np.mean(coefs_l2, axis=0)
    coef_l1 = np.mean(coefs_l1, axis=0)

    oof_ba_l2 = float(balanced_accuracy_score(y_sub, np.argmax(oof_l2, axis=1)))
    oof_ba_l1 = float(balanced_accuracy_score(y_sub, np.argmax(oof_l1, axis=1)))
    rec_l2 = recall_score(y_sub, np.argmax(oof_l2, axis=1), average=None, labels=[0, 1, 2])
    rec_l1 = recall_score(y_sub, np.argmax(oof_l1, axis=1), average=None, labels=[0, 1, 2])

    print(f"\n{'='*65}")
    print(f"  OOF balanced accuracy (subsample, {N_SPLITS}-fold)")
    print(f"{'='*65}")
    print(f"  L2 (C={C_L2}): {oof_ba_l2:.4f}  recall={dict(zip(CLASSES, rec_l2.round(4)))}")
    print(f"  L1 (C={C_L1}): {oof_ba_l1:.4f}  recall={dict(zip(CLASSES, rec_l1.round(4)))}")

    n_l1_zero = int((np.abs(coef_l1).max(axis=0) < 1e-6).sum())
    print(f"  L1 zeroed {n_l1_zero}/{len(feat_names)} features "
          f"({100*n_l1_zero/len(feat_names):.1f}%)")

    print_top_features(coef_l2, feat_names, f"L2 LR (C={C_L2})")
    print_top_features(coef_l1, feat_names, f"L1 LR (C={C_L1})")
    print_extra_section(coef_l2, coef_l1, feat_names, extra_cols)
    print_swallowed(coef_l2, coef_l1, feat_names, top_k=25)

    # Group-level importance summary
    print(f"\n{'='*65}")
    print(f"  GROUP SUMMARY — mean |L2 coef| by feature group")
    print(f"{'='*65}")
    from collections import defaultdict
    group_l2: dict[str, list[float]] = defaultdict(list)
    for i, n in enumerate(feat_names):
        group_l2[_group(n)].append(float(np.abs(coef_l2[:, i]).max()))
    rows = [(g, np.mean(v), np.max(v), len(v)) for g, v in group_l2.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"  {'Group':12s}  {'mean|coef|':>10}  {'max|coef|':>10}  {'n_feat':>7}")
    for g, mean_c, max_c, n in rows:
        print(f"  {g:12s}  {mean_c:10.4f}  {max_c:10.4f}  {n:7d}")


if __name__ == "__main__":
    main()
