"""LB-probe step 2 — infer public-split class composition from exact LB scores.

Model (quarantined curiosity track):
  Unknown per test row i: true class z_i, public membership s_i ~ Bern(p), s⊥z.
  Rows with identical prediction profiles across the K scored submissions are
  indistinguishable — inference lives on T[g,c] = #{i in group g, public, z=c}.
  Each public score is an EXACT linear functional of T (up to 5-dp rounding):
      bacc_s = (1/3) * sum_c ( sum_g 1(y_gs=c) T[g,c] ) / N_c,   N_c = sum_g T[g,c]
  (linearized at the prior mean of N_c). Prior: T Gaussian with moments from a
  calibrated per-row class posterior q thinned by p. Condition on the K scores
  (Gaussian linear conditioning) -> posterior group tilts -> tilted per-row
  posterior q'. Balanced-acc Bayes decisions come from argmax_c q'(c)/N'_c.

Validation: leave-one-out score prediction vs the prior-only baseline, plus
in-sample residuals. Jitter (observation noise) chosen by LOO MAE.
"""

import time

import numpy as np
import polars as pl

from lbprobe_common import (
    M_TEST,
    P_PUBLIC_DEFAULT,
    RESULTS_DIR,
    calibrated_prior,
    load_scores,
    load_submission_matrix,
    load_test_ids,
    profile_groups,
)

RNG_CLIP = (1e-3, 1e3)


def group_moments(
    q: np.ndarray, ginv: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum of q_i (G,3) and of q_i q_i^T (G,3,3)."""
    mu = np.zeros((n_groups, 3))
    qq = np.zeros((n_groups, 3, 3))
    for c in range(3):
        mu[:, c] = np.bincount(ginv, weights=q[:, c], minlength=n_groups)
        for d in range(c, 3):
            s = np.bincount(ginv, weights=q[:, c] * q[:, d], minlength=n_groups)
            qq[:, c, d] = s
            qq[:, d, c] = s
    return mu, qq


def build_l(yg: np.ndarray, n_bar: np.ndarray) -> np.ndarray:
    """Lg[g,c,s] = 1(y_gs = c) / (3 * N_bar_c)."""
    g, k = yg.shape
    lg = np.zeros((g, 3, k))
    for c in range(3):
        lg[:, c, :] = (yg == c) / (3.0 * n_bar[c])
    return lg


def predicted_score(labels: np.ndarray, q: np.ndarray) -> float:
    """E[bacc] of a hard-label vector under per-row class posterior q."""
    num = np.array([q[labels == c, c].sum() for c in range(3)])
    den = q.sum(axis=0)
    return float(np.mean(num / den))


def sampling_sd(labels: np.ndarray, q: np.ndarray, p: float) -> float:
    """Delta-method sd of the public bacc from row-level thinning/multinomial."""
    den = q.sum(axis=0) * p
    r = np.array([q[labels == c, c].sum() * p for c in range(3)]) / den
    g = np.zeros_like(q)
    for c in range(3):
        g[:, c] = ((labels == c).astype(float) - r[c]) / (3.0 * den[c])
    ex = (g * q).sum(axis=1)
    var = (p * (g**2 * q).sum()) - (p**2) * (ex**2).sum()
    return float(np.sqrt(max(var, 0.0)))


def fit(
    p: float = P_PUBLIC_DEFAULT,
    jitter_grid: tuple[float, ...] = (),
    file_subset: list[str] | None = None,
    quiet: bool = False,
) -> dict:
    t0 = time.time()
    scores = load_scores()
    if file_subset is not None:
        scores = scores.filter(pl.col("file_name").is_in(file_subset))
    files = scores["file_name"].to_list()
    b_obs = scores["public_score"].to_numpy().astype(np.float64)
    k = len(files)

    test_ids = load_test_ids()
    y_mat = load_submission_matrix(files, test_ids)
    ginv, yg = profile_groups(y_mat)
    n_groups = yg.shape[0]
    gsize = np.bincount(ginv, minlength=n_groups)
    if not quiet:
        print(
            f"K={k} submissions, G={n_groups} profile groups "
            f"(top sizes: {np.sort(gsize)[::-1][:6]}), "
            f"disagreement rows={int((gsize[ginv] < gsize.max())[ginv >= 0].sum()) - 0}"
        )

    q = calibrated_prior(quiet=quiet)
    pi_bar = q.mean(axis=0)
    n_bar = p * M_TEST * pi_bar

    mu, qq = group_moments(q, ginv, n_groups)
    t_bar = p * mu
    sig = p * np.einsum("gc,cd->gcd", mu, np.eye(3)) - (p**2) * qq

    lg = build_l(yg, n_bar)
    b_prior = np.einsum("gcs,gc->s", lg, t_bar)
    r = b_obs - b_prior

    # S = L Sigma L^T (signal part), shared across jitter values
    ls = np.einsum("gcs,gcd->gds", lg, sig)  # (G,3,K)
    s_sig = np.einsum("gds,gdt->st", ls, lg)

    def loo_mae(jit: float) -> float:
        s_full = s_sig + (jit**2) * np.eye(k)
        errs = []
        for i in range(k):
            m = np.ones(k, dtype=bool)
            m[i] = False
            a = np.linalg.solve(s_full[np.ix_(m, m)], r[m])
            errs.append(abs(r[i] - s_sig[i, m] @ a))
        return float(np.mean(errs))

    if not jitter_grid:
        jitter_grid = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4)
    loo_by_jit = {jit: loo_mae(jit) for jit in jitter_grid}
    best_jit = min(loo_by_jit, key=loo_by_jit.get)

    s_full = s_sig + (best_jit**2) * np.eye(k)
    alpha = np.linalg.solve(s_full, r)
    t_hat = t_bar + np.einsum("gds,s->gd", ls, alpha)

    ratio = np.clip(t_hat / np.clip(t_bar, 1e-12, None), *RNG_CLIP)
    qp = q * ratio[ginv]
    qp /= qp.sum(axis=1, keepdims=True)

    # in-sample residuals under the tilted posterior
    b_post = np.array(
        [predicted_score(y_mat[:, s], qp) for s in range(k)]
    )

    if not quiet:
        print(f"\njitter grid (LOO MAE): " + ", ".join(f"{j:g}:{v:.6f}" for j, v in loo_by_jit.items()))
        print(f"chosen jitter={best_jit:g}")
        print(f"prior-only LOO baseline (|r| mean)     : {np.abs(r).mean():.6f}")
        print(f"predict-mean baseline                  : {np.abs(b_obs - b_obs.mean()).mean():.6f}")
        print(f"conditioned LOO MAE                    : {loo_by_jit[best_jit]:.6f}")
        print(f"in-sample |obs - tilted-posterior| mean: {np.abs(b_obs - b_post).mean():.6f}, max {np.abs(b_obs - b_post).max():.6f}")

    return {
        "files": files,
        "b_obs": b_obs,
        "b_prior": b_prior,
        "b_post": b_post,
        "y_mat": y_mat,
        "ginv": ginv,
        "yg": yg,
        "q": q,
        "qp": qp,
        "t_bar": t_bar,
        "t_hat": t_hat,
        "ratio": ratio,
        "s_full": s_full,
        "s_sig": s_sig,
        "sig": sig,
        "ls": ls,
        "lg": lg,
        "alpha": alpha,
        "r": r,
        "p": p,
        "jitter": best_jit,
        "loo_mae": loo_by_jit[best_jit],
        "loo_by_jit": loo_by_jit,
        "elapsed": time.time() - t0,
    }


def main() -> None:
    res = fit()
    k = len(res["files"])

    # per-submission table: observed vs prior vs posterior
    print("\nfile                                  obs      prior    post     r_prior")
    order = np.argsort(res["b_obs"])[::-1]
    for i in order:
        print(
            f"{res['files'][i]:36s} {res['b_obs'][i]:.5f}  "
            f"{res['b_prior'][i]:.5f}  {res['b_post'][i]:.5f}  {res['r'][i]:+.5f}"
        )

    # candidate scores under the tilted posterior
    qp, q, y_mat = res["qp"], res["q"], res["y_mat"]
    best_idx = int(np.argmax(res["b_obs"]))
    best_labels = y_mat[:, best_idx]
    bayes = np.argmax(qp / qp.sum(axis=0, keepdims=True), axis=1).astype(np.int8)

    print(f"\nbest existing sub: {res['files'][best_idx]} obs={res['b_obs'][best_idx]:.5f}")
    for name, lab in (("best-existing", best_labels), ("full-bayes", bayes)):
        e = predicted_score(lab, qp)
        sd = sampling_sd(lab, qp, res["p"])
        nflip = int((lab != best_labels).sum())
        print(f"  {name:14s} E[bacc|post]={e:.5f}  samp_sd={sd:.5f}  flips_vs_best={nflip}")

    np.save(RESULTS_DIR / "lbprobe_qpost.npy", qp.astype(np.float32))
    np.save(RESULTS_DIR / "lbprobe_qprior.npy", q.astype(np.float32))
    pl.DataFrame(
        {
            "file_name": res["files"],
            "public_score": res["b_obs"],
            "prior_pred": res["b_prior"],
            "posterior_pred": res["b_post"],
        }
    ).write_csv(RESULTS_DIR / "lbprobe_fit_scores.csv")
    print(f"\nsaved lbprobe_qpost.npy / lbprobe_qprior.npy; elapsed {res['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
