"""LB-probe step 3 — build the public-LB-optimized submission (curiosity track).

Candidates:
  full-bayes : argmax_c q'(c)/N'_c under the tilted posterior (all rows).
  margin-k   : start from the best-scoring existing submission, apply only the
               k flips with the largest expected per-row gain (winner's-curse
               control — fewer, higher-confidence flips).

For each candidate vs the best existing submission we report the posterior
expected delta on the SAME public split, with two uncertainty terms:
  samp_sd : which flip rows are public / their true classes (thinning noise)
  tilt_sd : posterior uncertainty of the group compositions (what the K scores
            could not pin down), via Gaussian conditioning covariance.
"""

import numpy as np
import polars as pl

from lbprobe_common import SUBS_DIR, decode_labels, load_test_ids
from lbprobe_fit import fit, predicted_score

TARGET = "health_condition"


def group_shares(
    labels: np.ndarray, qp: np.ndarray, ginv: np.ndarray, n_groups: int
) -> np.ndarray:
    """share[g,c] = posterior fraction of group-g class-c mass assigned label c."""
    num = np.zeros((n_groups, 3))
    den = np.zeros((n_groups, 3))
    for c in range(3):
        den[:, c] = np.bincount(ginv, weights=qp[:, c], minlength=n_groups)
        w = qp[:, c] * (labels == c)
        num[:, c] = np.bincount(ginv, weights=w, minlength=n_groups)
    return num / np.clip(den, 1e-12, None)


def delta_uncertainty(
    lab_a: np.ndarray, lab_b: np.ndarray, res: dict
) -> tuple[float, float]:
    """(samp_sd, tilt_sd) of bacc(A) - bacc(B) on the same public split."""
    qp, p, ginv = res["qp"], res["p"], res["ginv"]
    n_groups = res["yg"].shape[0]
    n_bar_post = qp.sum(axis=0) * p

    flip = lab_a != lab_b
    g_row = np.zeros((flip.sum(), 3))
    qf = qp[flip]
    for c in range(3):
        g_row[:, c] = (
            (lab_a[flip] == c).astype(float) - (lab_b[flip] == c).astype(float)
        ) / (3.0 * n_bar_post[c])
    ex = (g_row * qf).sum(axis=1)
    var_samp = p * (g_row**2 * qf).sum() - (p**2) * (ex**2).sum()

    # tilt term: project the delta functional onto group-composition space
    sh_a = group_shares(lab_a, qp, ginv, n_groups)
    sh_b = group_shares(lab_b, qp, ginv, n_groups)
    d = (sh_a - sh_b) / (3.0 * n_bar_post)  # (G,3), l_A - l_B
    sig, lg, s_full = res["sig"], res["lg"], res["s_full"]
    sig_d = np.einsum("gcd,gd->gc", sig, d)
    d_sig_d = float(np.einsum("gc,gc->", d, sig_d))
    v = np.einsum("gcs,gc->s", lg, sig_d)
    var_tilt = d_sig_d - float(v @ np.linalg.solve(s_full, v))
    return float(np.sqrt(max(var_samp, 0.0))), float(np.sqrt(max(var_tilt, 0.0)))


def main() -> None:
    res = fit()
    qp, q, y_mat, ginv = res["qp"], res["q"], res["y_mat"], res["ginv"]
    files, b_obs, p = res["files"], res["b_obs"], res["p"]

    best_idx = int(np.argmax(b_obs))
    best_labels = y_mat[:, best_idx]
    best_name, best_score = files[best_idx], b_obs[best_idx]

    n_post = qp.sum(axis=0)
    value = qp / n_post  # per-row per-class expected bacc contribution (x 3/p M)
    bayes = np.argmax(value, axis=1).astype(np.int8)

    # expected per-row gain of flipping best -> bayes
    rows = np.arange(len(bayes))
    gain = value[rows, bayes] - value[rows, best_labels]
    flip_mask = bayes != best_labels
    order = np.argsort(gain)[::-1]

    candidates: dict[str, np.ndarray] = {"full_bayes": bayes}
    for k in (300, 700):
        lab = best_labels.copy()
        take = order[:k]
        take = take[gain[take] > 0]
        lab[take] = bayes[take]
        candidates[f"margin_{k}"] = lab

    base_post = predicted_score(best_labels, qp)
    base_prior = predicted_score(best_labels, q)
    print(
        f"\nreference: {best_name} obs={best_score:.5f} "
        f"E_post={base_post:.5f} E_prior={base_prior:.5f}"
    )
    print(
        "\ncandidate     flips  E_post   Δpost     E_prior  Δprior    samp_sd  tilt_sd  z(Δ/sd)"
    )
    summary = {}
    for name, lab in candidates.items():
        e_post = predicted_score(lab, qp)
        e_prior = predicted_score(lab, q)
        d_post = e_post - base_post
        d_prior = e_prior - base_prior
        ssd, tsd = delta_uncertainty(lab, best_labels, res)
        sd = float(np.hypot(ssd, tsd))
        z = d_post / sd if sd > 0 else float("inf")
        nflip = int((lab != best_labels).sum())
        print(
            f"{name:12s} {nflip:6d}  {e_post:.5f} {d_post:+.5f}  "
            f"{e_prior:.5f} {d_prior:+.5f}  {ssd:.5f}  {tsd:.5f}  {z:5.1f}"
        )
        summary[name] = (lab, e_post, d_post, z, nflip)

    # flip anatomy for full-bayes
    fb = candidates["full_bayes"]
    print("\nfull-bayes flip anatomy (from -> to, count, mean expected row gain x1e6):")
    for c_from in range(3):
        for c_to in range(3):
            m = (best_labels == c_from) & (fb == c_to) & (c_from != c_to)
            if m.sum():
                mg = gain[m].mean() * 1e6
                print(f"  {c_from}->{c_to}: {int(m.sum()):5d}  gain/row {mg:8.2f}")

    test_ids = load_test_ids()
    for name, (lab, e_post, d_post, z, nflip) in summary.items():
        out = SUBS_DIR / f"lbprobe_{name}.csv"
        pl.DataFrame({"id": test_ids, TARGET: decode_labels(lab)}).write_csv(out)
        print(f"wrote {out.name}  (E_post {e_post:.5f}, Δ {d_post:+.5f}, z {z:.1f})")


if __name__ == "__main__":
    main()
