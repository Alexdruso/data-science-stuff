"""Feature engineering for PS S6E6 — Predicting Stellar Class.

Single source of truth for all feature transforms. Every training script and
notebook MUST import from here — never define transforms elsewhere.

Leakage rules (see CLAUDE.md for full details):
- build_features() touches only raw input columns, never the target.
- compute_group_features() takes train_raw as its reference; stats are global
  priors computed on all training rows (low-leakage). Target-encoding that
  is fold-aware must be done in the training script per fold, not here.
- All frames sorted by SORT_KEY so oof/test .npy arrays stay aligned.
"""

import math

import numpy as np
import polars as pl

TARGET = "class"
SORT_KEY: list[str] = ["id"]

# Columns excluded from the feature matrix (id + target only).
# Add string categoricals that hurt gradient boosters to EXCLUDE_COLS.
EXCLUDE_COLS: frozenset[str] = frozenset({TARGET, "id"})


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sort and add engineered features. Safe to call on train or test.

    Must never read or depend on TARGET — test has no labels.
    """
    df = df.sort(SORT_KEY)

    # SDSS photometric color indices — strong class discriminators
    df = df.with_columns([
        (pl.col("u") - pl.col("g")).alias("u_g"),
        (pl.col("g") - pl.col("r")).alias("g_r"),
        (pl.col("r") - pl.col("i")).alias("r_i"),
        (pl.col("i") - pl.col("z")).alias("i_z"),
        (pl.col("u") - pl.col("z")).alias("u_z"),  # full-range blue/red
        # log1p handles the small negative redshifts (-0.01) from measurement noise
        (pl.col("redshift") + 1.0).log(base=math.e).alias("log1p_redshift"),
    ])

    return df


def add_physics_features(df: pl.DataFrame) -> pl.DataFrame:
    """Physics-motivated nonlinear features for the low-z STAR/GALAXY boundary.

    The incumbent errors concentrate where stars (redshift ≈ 0) overlap low-z
    galaxies. Distance modulus μ ≈ 5·log10(redshift) (∝ luminosity distance at low
    z); the per-band absolute-magnitude proxy m − μ is a nonlinear redshift×
    magnitude combination that axis-aligned GBDT splits approximate poorly — which
    is why ~1000 trees on the raw bands plateau. For stars redshift ≈ 0 so μ is
    undefined → null, itself a strong "local object" signal.

    Experimental: applied by experiment scripts, not yet folded into
    build_features() (would invalidate existing oof/test .npy). Fold in + regenerate
    all models only if it clears the 0.9654 argmax gate.
    """
    dist_mod = (
        pl.when(pl.col("redshift") > 0)
        .then(5.0 * pl.col("redshift").log10())
        .otherwise(None)
    )
    return df.with_columns([
        dist_mod.alias("dist_mod"),
        *[(pl.col(b) - dist_mod).alias(f"absmag_{b}") for b in ["u", "g", "r", "i", "z"]],
    ])


def add_deotte_features(df: pl.DataFrame) -> pl.DataFrame:
    """High-gain features from cdeotte's XGB v1 (CV 0.9669) — run AFTER build_features
    (needs the u_g/g_r/r_i/i_z colors).

    The top of his importance list is dominated by redshift×band interactions and
    band/redshift ratios — NOT plain colors (which we already had and which were
    flat). Plus magnitude aggregations, SED-shape polynomials, redshift transforms,
    and color-plane polar coords. SDSS17 priors + quantile-bin target encoding are
    deferred to a fold-aware step (leak-sensitive).
    """
    bands = ["u", "g", "r", "i", "z"]
    eps = 1e-6
    df = df.with_columns(
        [(pl.col("redshift") * pl.col(b)).alias(f"redshift_{b}") for b in bands]
        + [(pl.col(b) / (pl.col("redshift").abs() + eps)).alias(f"{b}_over_redshift")
           for b in bands]
    )
    mag_mean = pl.mean_horizontal([pl.col(b) for b in bands])
    sq_mean = pl.mean_horizontal([pl.col(b) ** 2 for b in bands])
    df = df.with_columns([
        mag_mean.alias("mag_mean"),
        pl.min_horizontal([pl.col(b) for b in bands]).alias("mag_min"),
        pl.max_horizontal([pl.col(b) for b in bands]).alias("mag_max"),
        (sq_mean - mag_mean ** 2).clip(lower_bound=0.0).sqrt().alias("mag_std"),
        # SED shape: least-squares slope over bands 0..4 (x-mean = [-2,-1,0,1,2])
        ((-2 * pl.col("u") - pl.col("g") + pl.col("i") + 2 * pl.col("z")) / 10.0).alias("mag_slope"),
        (pl.col("u") - 2 * pl.col("r") + pl.col("z")).alias("mag_curvature"),
        (pl.col("u") - 2 * pl.col("g") + pl.col("r")).alias("blue_curvature"),
        (pl.col("r") - 2 * pl.col("i") + pl.col("z")).alias("red_curvature"),
        pl.col("redshift").abs().alias("redshift_abs"),
        (pl.col("redshift").abs() + 1.0).log().alias("redshift_log1p_abs"),
        (pl.col("redshift") < 0).cast(pl.Int8).alias("redshift_is_neg"),
    ])
    df = df.with_columns([
        (pl.col("mag_max") - pl.col("mag_min")).alias("mag_range"),
        (pl.col("u_g") ** 2 + pl.col("g_r") ** 2).sqrt().alias("cpr_ug_gr"),
        pl.arctan2(pl.col("u_g"), pl.col("g_r") + eps).alias("cpa_ug_gr"),
        (pl.col("r_i") ** 2 + pl.col("i_z") ** 2).sqrt().alias("cpr_ri_iz"),
        pl.arctan2(pl.col("r_i"), pl.col("i_z") + eps).alias("cpa_ri_iz"),
    ])
    return df


# ── SDSS17 class-prior features (cdeotte's XGB lever) ─────────────────────────
# Quantile-bin selected features, then join the per-bin class RATE measured on the
# REAL SDSS17 dataset, smoothed toward the original global prior. These are a
# deterministic function of a row's own features + a fixed EXTERNAL table → leak-safe
# (the encoding source is the original data, never the train/val being predicted) and
# applied identically to train/val/test, so any signal transfers to the LB.
SDSS17_BIN_FEATURES: list[str] = ["redshift", "u_g", "g_r", "r_i", "i_z"]
SDSS17_NBINS = 64
SDSS17_CROSSES: list[tuple[str, str]] = [("redshift", "u_g"), ("redshift", "g_r")]
SDSS17_CROSS_NBINS = 24
SDSS17_SMOOTH = 20.0
_PRIOR_CLASSES: list[str] = ["GALAXY", "QSO", "STAR"]


def _bin_codes(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Map values to integer bin codes in [0, len(breaks)] via the given cut edges."""
    return np.searchsorted(breaks, values, side="right").astype(np.int64)


def _codes_for(
    orig_vals: np.ndarray, df_vals: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin edges from the ORIGINAL feature distribution; apply to both."""
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    breaks = np.unique(np.quantile(orig_vals, qs))
    return _bin_codes(orig_vals, breaks), _bin_codes(df_vals, breaks)


def _attach_priors(
    out: dict[str, np.ndarray],
    name: str,
    orig_codes: np.ndarray,
    df_codes: np.ndarray,
    onehot: dict[str, np.ndarray],
    g_prior: dict[str, float],
    smooth: float = SDSS17_SMOOTH,
) -> None:
    """Per-bin original count + smoothed per-class rate, mapped onto df rows."""
    nbins = int(max(orig_codes.max(), df_codes.max())) + 1
    count = np.bincount(orig_codes, minlength=nbins).astype(np.float64)
    out[f"orig_{name}_count"] = np.log1p(count[df_codes])
    for c in _PRIOR_CLASSES:
        csum = np.bincount(orig_codes, weights=onehot[c], minlength=nbins)
        prior = (csum + smooth * g_prior[c]) / (count + smooth)
        out[f"orig_{name}_prior_{c}"] = prior[df_codes]


def add_sdss17_priors(df: pl.DataFrame, orig: pl.DataFrame) -> pl.DataFrame:
    """Join class-prior columns learned from the REAL SDSS17 data, per quantile bin.

    Run AFTER build_features (needs the u_g/g_r/r_i/i_z colors). `orig` is the raw
    SDSS17 CSV (fedesoriano/stellar-classification-dataset-sdss17). The synthetic
    generator preserves P(class | features) (we classify at 96%), so predict this is
    largely flat — but it is the last untried piece of cdeotte's XGB FE (his XGB hit
    CV 0.9669). Gate on low-z STAR recall + argmax, then re-stack; do not iterate.
    """
    orig = orig.filter(pl.col("u") > -1000.0).with_columns([  # drop u=g=z=-9999 row
        (pl.col("u") - pl.col("g")).alias("u_g"),
        (pl.col("g") - pl.col("r")).alias("g_r"),
        (pl.col("r") - pl.col("i")).alias("r_i"),
        (pl.col("i") - pl.col("z")).alias("i_z"),
    ])
    y = orig["class"].to_numpy()
    onehot = {c: (y == c).astype(np.float64) for c in _PRIOR_CLASSES}
    g_prior = {c: float(onehot[c].mean()) for c in _PRIOR_CLASSES}

    out: dict[str, np.ndarray] = {}
    for f in SDSS17_BIN_FEATURES:
        oc, dc = _codes_for(orig[f].to_numpy(), df[f].to_numpy(), SDSS17_NBINS)
        _attach_priors(out, f, oc, dc, onehot, g_prior)
    for fa, fb in SDSS17_CROSSES:
        oca, dca = _codes_for(orig[fa].to_numpy(), df[fa].to_numpy(), SDSS17_CROSS_NBINS)
        ocb, dcb = _codes_for(orig[fb].to_numpy(), df[fb].to_numpy(), SDSS17_CROSS_NBINS)
        m = SDSS17_CROSS_NBINS + 1
        _attach_priors(out, f"{fa}_x_{fb}", oca * m + ocb, dca * m + dcb, onehot, g_prior)

    return df.with_columns([pl.Series(k, v) for k, v in out.items()])


def compute_group_features(
    train_raw: pl.DataFrame,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Join pre-computed group statistics from train_raw onto df.

    train_raw is the full raw training CSV — never a fold slice.
    df may be train or test; it is already sorted by build_features().

    ⚠️  These stats use TARGET from train_raw. That is intentional: they are
    global priors, not fold-specific. For fold-aware target encoding use
    sklearn TargetEncoder with cv=5 inside the training loop instead.
    """
    # ── placeholder: add group aggregates here after EDA ──────────────────
    # Example pattern:
    #   global_rate = float(train_raw[TARGET].value_counts(normalize=True)...)
    #   group_stat = (
    #       train_raw.group_by("some_cat")
    #       .agg(...)
    #   )
    #   df = df.join(group_stat, on="some_cat", how="left")
    # ──────────────────────────────────────────────────────────────────────

    return df
