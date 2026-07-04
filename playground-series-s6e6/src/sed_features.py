"""Physics prior: SED template-fit chi2 features from SDSS's OWN classification eigenspectra.

The competition labels descend from SDSS's spec1d pipeline, which classifies each object by
fitting STAR / GALAXY / QSO spectral templates and taking the best chi2. We don't have spectra,
but we have ugriz photometry + spectroscopic redshift, so we do the photometric analog:

  for each object, redshift each class's template(s) to the object's z (stars: z=0), integrate
  through the real SDSS ugriz filter curves (speclite), fit an amplitude (color-shape fit,
  1/flux^2 weighting), and record the best chi2 per class.

Templates = SDSS idlspec2d eigenspectra (spEigenStar 123 stellar templates; spEigenGal /
spEigenQSO mean eigenspectrum = mean galaxy / QSO SED, both all-positive). This is the most
faithful possible physics prior — the exact basis the labels were assigned with. The chi2 features
are a deterministic, target-free, leak-safe function of (u,g,r,i,z,redshift) → no fold-awareness.

PRE-FLIGHT (the make-or-break guard): argmin(chi2) must classify train WELL above the 65.4%
majority baseline. These are the literal label templates — if it doesn't, it's a units/integration/
z bug, NOT a physics verdict. Also reports chi2 correlation with redshift+colors (near-deterministic
=> predicts flat before training).

Run:  python src/sed_features.py        # builds + caches features, prints sanity check
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from speclite import filters

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
TEMPL_DIR = DATA_DIR / "templates"
BANDS = ["u", "g", "r", "i", "z"]
AB_OFF = np.array([-0.04, 0.0, 0.0, 0.0, 0.02])  # SDSS asinh mags -> AB
ZGRID = np.arange(0.0, 7.111, 0.01)
TEMPL = {"star": "spEigenStar-55734", "gal": "spEigenGal-56436", "qso": "spEigenQSO-55732"}
CLASS_TO_INT = {"GALAXY": 0, "QSO": 1, "STAR": 2}


def filter_arrays() -> list[tuple[np.ndarray, np.ndarray]]:
    sd = filters.load_filters("sdss2010-*")
    return [(np.asarray(f.wavelength, float), np.asarray(f.response, float)) for f in sd]


def template_grid(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lam_rest, flux[n_templates, n_pix])."""
    with fits.open(TEMPL_DIR / f"{TEMPL[name]}.fits") as h:
        hd = h[0].header
        flux = np.atleast_2d(np.asarray(h[0].data, float))
        n = hd["NAXIS1"]
        lam = 10 ** (hd["COEFF0"] + hd["COEFF1"] * np.arange(n))
    return lam, flux


def synth_maggies(lam_rest, flux_rows, z, farr):
    """AB maggies of each template row at redshift z through each band; + coverage mask.

    maggie_b ∝ ∫ T_λ(λ_obs) R_b dλ / ∫ R_b/λ² dλ  (amplitude-free AB normalization).
    Returns (maggies[n_rows, 5], covered[5]).
    """
    lam_obs = lam_rest * (1.0 + z)
    n = flux_rows.shape[0]
    out = np.zeros((n, 5))
    cov = np.zeros(5, bool)
    for b, (lam, R) in enumerate(farr):
        supp = lam[R > 1e-3]
        cov[b] = (supp.min() >= lam_obs.min()) and (supp.max() <= lam_obs.max())
        den = np.trapezoid(R / lam**2, lam)
        # interp each template row onto filter grid (0 outside template range)
        Ti = np.vstack([np.interp(lam, lam_obs, flux_rows[k], left=0.0, right=0.0) for k in range(n)])
        out[:, b] = np.trapezoid(Ti * R, lam, axis=1) / den
    return out, cov


def chi2_amplitude(f_obs, T, cov):
    """Color-shape chi2 (1/flux² weight, free amplitude) of obs flux [N,5] vs templates.

    T: [M,5] template maggies, cov: [5] band mask. r=T/f_obs; a=Σr/Σr²; chi2=Σ(1-a·r)².
    Returns chi2 [N, M] over covered bands (needs ≥2 covered).
    """
    cb = np.where(cov)[0]
    fo = f_obs[:, cb]                       # [N, C]
    Tc = T[:, cb]                           # [M, C]
    r = Tc[None, :, :] / fo[:, None, :]     # [N, M, C]
    sr = r.sum(2)
    srr = (r * r).sum(2)
    a = sr / np.clip(srr, 1e-30, None)      # [N, M]
    chi2 = ((1.0 - a[:, :, None] * r) ** 2).sum(2)  # [N, M]
    return chi2


def build_features(df: pd.DataFrame, farr, star, gal, qso) -> np.ndarray:
    m = df[BANDS].to_numpy(float) + AB_OFF
    f_obs = 10 ** (-0.4 * m)                # AB maggies [N,5]
    z = np.clip(df["redshift"].to_numpy(float), 0.0, ZGRID[-1])
    zi = np.rint(z / 0.01).astype(int)
    N = len(df)
    out = np.full((N, 3), np.nan)           # chi2_star, chi2_gal, chi2_qso

    # STAR: 123 templates at z=0, same for all objects (chunk rows for memory)
    star_mag, star_cov = star
    for s in range(0, N, 50_000):
        out[s:s + 50_000, 0] = chi2_amplitude(f_obs[s:s + 50_000], star_mag, star_cov).min(1)

    # GAL / QSO: mean-SED eigenspectrum at object z (group by z-bin)
    gal_mag, gal_cov = gal       # [nz,5], [nz,5]
    qso_mag, qso_cov = qso
    for zb in np.unique(zi):
        idx = np.where(zi == zb)[0]
        fo = f_obs[idx]
        if gal_cov[zb].sum() >= 2:
            out[idx, 1] = chi2_amplitude(fo, gal_mag[zb][None, :], gal_cov[zb])[:, 0]
        if qso_cov[zb].sum() >= 2:
            out[idx, 2] = chi2_amplitude(fo, qso_mag[zb][None, :], qso_cov[zb])[:, 0]
    return np.log10(np.clip(out, 1e-12, None))  # log chi2 — heavy-tailed


def precompute(farr):
    lam_s, flux_s = template_grid("star")
    star = synth_maggies(lam_s, flux_s, 0.0, farr)                      # [123,5],[5]
    lam_g, flux_g = template_grid("gal")
    lam_q, flux_q = template_grid("qso")
    nz = len(ZGRID)
    gal_mag = np.zeros((nz, 5)); gal_cov = np.zeros((nz, 5), bool)
    qso_mag = np.zeros((nz, 5)); qso_cov = np.zeros((nz, 5), bool)
    for i, z in enumerate(ZGRID):
        mg, cg = synth_maggies(lam_g, flux_g[:1], z, farr)
        gal_mag[i], gal_cov[i] = mg[0], cg
        mq, cq = synth_maggies(lam_q, flux_q[:1], z, farr)
        qso_mag[i], qso_cov[i] = mq[0], cq
    return star, (gal_mag, gal_cov), (qso_mag, qso_cov)


def main() -> None:
    farr = filter_arrays()
    print("Precomputing template maggies on z-grid ...")
    star, gal, qso = precompute(farr)

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values("id").reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values("id").reset_index(drop=True)
    print("Building SED features (train) ...")
    ftr = build_features(train, farr, star, gal, qso)
    print("Building SED features (test) ...")
    fte = build_features(test, farr, star, gal, qso)
    np.save(RESULTS_DIR / "sed_train.npy", ftr.astype(np.float32))
    np.save(RESULTS_DIR / "sed_test.npy", fte.astype(np.float32))

    # ---- PRE-FLIGHT SANITY CHECK ----
    y = train["class"].map(CLASS_TO_INT).to_numpy()
    valid = ~np.isnan(ftr).any(1)
    # argmin chi2 -> class. column order [star,gal,qso] -> map to {GAL0,QSO1,STAR2}
    col_to_class = np.array([2, 0, 1])  # star_col->STAR(2), gal_col->GAL(0), qso_col->QSO(1)
    pred = col_to_class[np.argmin(ftr[valid], axis=1)]
    yv = y[valid]
    from sklearn.metrics import balanced_accuracy_score, recall_score
    acc = (pred == yv).mean()
    bal = balanced_accuracy_score(yv, pred)
    rec = recall_score(yv, pred, average=None, labels=[0, 1, 2])
    print(f"\n=== PRE-FLIGHT (argmin-chi2 classifier) on {valid.mean()*100:.1f}% valid rows ===")
    print(f"  raw acc {acc:.4f}  balanced {bal:.4f}  [majority baseline 0.654 / chance 0.333]")
    print(f"  per-class recall GAL {rec[0]:.3f} QSO {rec[1]:.3f} STAR {rec[2]:.3f}")
    # correlation of chi2 feats with redshift + colors
    z = train["redshift"].to_numpy(float)
    cols = {"redshift": z, "u_g": (train.u - train.g).to_numpy(),
            "g_r": (train.g - train.r).to_numpy()}
    print("  |corr| of log-chi2 feats vs:")
    for nm, v in cols.items():
        cc = [abs(np.corrcoef(ftr[valid, j], v[valid])[0, 1]) for j in range(3)]
        print(f"    {nm:9s} star {cc[0]:.2f} gal {cc[1]:.2f} qso {cc[2]:.2f}")
    print(f"\n  nan rows: train {(~valid).sum()} ({(~valid).mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
