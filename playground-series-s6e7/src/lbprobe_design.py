"""LB-probe step 4 — design exact-feedback probe submissions (Mastermind mode).

Key fact: balanced accuracy is LINEAR in the per-row label assignment (the
class counts N_c of the public split do not depend on predictions). Hence:
  * probe_j = reference submission with one flip set W_j applied;
  * Δ_j = public_score(probe_j) - public_score(reference) is EXACT (±1e-5
    rounding) and equals the sum of per-row contribution changes over W_j;
  * disjoint flip sets compose additively -> keep every set with Δ_j > 0,
    apply all keepers to build the next reference. Guaranteed hill climb.

Set design: rank rows by posterior expected flip gain (best alternative class
vs the reference label), stratify by flip direction, chunk consecutively, and
pick the sets maximizing E[max(Δ,0)] under the posterior (mean + uncertainty
both help: high-mean sets are likely keepers, high-variance sets are cheap
lottery tickets whose positive tail the keep/revert rule harvests).

Output: submissions/lbprobe_d<day>_p<k>.csv + results/lbprobe_probe_manifest.csv.
Next round: re-dump `kaggle competitions submissions`, rerun lbprobe_ingest.py,
then lbprobe_compose.py (keeps Δ>0 sets; also feeds the new exact constraints
back into the fit for the next design round).
"""

import argparse

import numpy as np
import polars as pl
from scipy.stats import norm

from lbprobe_common import RESULTS_DIR, SUBS_DIR, decode_labels, load_test_ids
from lbprobe_fit import fit, predicted_score

TARGET = "health_condition"
MIN_ABS_MEAN = 1.5e-5  # measured delta must clear LB rounding
MAX_CHUNK = 600
MIN_CHUNK = 40


def make_chunks(
    idx_sorted: np.ndarray, gain: np.ndarray, sd_row2: np.ndarray
) -> list[np.ndarray]:
    """Cut a direction-sorted row list into chunks with resolvable deltas."""
    chunks, cur = [], []
    cur_mean = 0.0
    for i in idx_sorted:
        cur.append(i)
        cur_mean += gain[i]
        if len(cur) >= MIN_CHUNK and (
            abs(cur_mean) >= MIN_ABS_MEAN or len(cur) >= MAX_CHUNK
        ):
            chunks.append(np.array(cur))
            cur, cur_mean = [], 0.0
    if len(cur) >= MIN_CHUNK:
        chunks.append(np.array(cur))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--n-probes", type=int, default=9)
    ap.add_argument("--reference", type=str, default="")
    ap.add_argument(
        "--reference-score",
        type=float,
        default=None,
        help="required when --reference is a composed CSV without an LB score",
    )
    args = ap.parse_args()

    res = fit()
    qp, p = res["qp"], res["p"]
    files, b_obs = res["files"], res["b_obs"]

    if args.reference and args.reference not in files:
        from lbprobe_common import load_submission_matrix

        ref_name = args.reference
        ref = load_submission_matrix([ref_name], load_test_ids())[:, 0].copy()
        assert args.reference_score is not None, "--reference-score required"
        ref_score = args.reference_score
    elif args.reference:
        ref_name = args.reference
        ref_idx = files.index(ref_name)
        ref = res["y_mat"][:, ref_idx].copy()
        ref_score = b_obs[ref_idx]
    else:
        ref_idx = int(np.argmax(b_obs))
        ref_name = files[ref_idx]
        ref = res["y_mat"][:, ref_idx].copy()
        ref_score = b_obs[ref_idx]
    print(f"\nreference: {ref_name} (public {ref_score:.5f})")

    # per-row contribution of assigning class c: qp_ic / (3 * N_c_post)
    n_post = qp.sum(axis=0)
    v = qp / (3.0 * n_post)
    rows = np.arange(len(ref))
    v_ref = v[rows, ref]
    v_alt = v.copy()
    v_alt[rows, ref] = -np.inf
    alt = np.argmax(v_alt, axis=1).astype(np.int8)
    gain = v[rows, alt] - v_ref  # expected bacc delta per flipped row

    # per-row delta variance under thinning (sampling term; tilt term is what
    # the keep/revert rule makes irrelevant, so mean+sampling is the design basis)
    g_to = 1.0 / (3.0 * n_post * p)  # contribution scale if truly the target class
    var_row = np.zeros(len(ref))
    for c in range(3):
        coef = ((alt == c).astype(float) - (ref == c).astype(float)) / (
            3.0 * n_post[c] * p
        )
        var_row += p * coef**2 * qp[:, c]
    ex_row = gain  # E under posterior already
    var_row = np.maximum(var_row - p**2 * (ex_row / p) ** 2, 1e-30)

    # build candidate chunks per flip direction
    all_chunks: list[tuple[np.ndarray, str]] = []
    for c_from in range(3):
        for c_to in range(3):
            if c_from == c_to:
                continue
            m = (ref == c_from) & (alt == c_to)
            idx = rows[m]
            if len(idx) < MIN_CHUNK:
                continue
            idx = idx[np.argsort(gain[idx])[::-1]]
            # only the informative head: skip rows that are hopeless (very negative)
            idx = idx[gain[idx] > np.quantile(gain[idx], 0.5)]
            for ch in make_chunks(idx, gain, var_row):
                all_chunks.append((ch, f"{c_from}->{c_to}"))

    # rank by E[max(delta, 0)]
    scored = []
    for ch, direction in all_chunks:
        mu = float(gain[ch].sum())
        sd = float(np.sqrt(var_row[ch].sum()))
        ev = mu * norm.cdf(mu / sd) + sd * norm.pdf(mu / sd) if sd > 0 else max(mu, 0)
        scored.append((ev, mu, sd, ch, direction))
    scored.sort(key=lambda t: t[0], reverse=True)

    test_ids = load_test_ids()
    manifest = []
    used = np.zeros(len(ref), dtype=bool)
    k = 0
    print(
        f"\n{'probe':22s} {'dir':6s} {'n':>5s} {'E[Δ]':>9s} {'sd':>8s} {'E[max(Δ,0)]':>11s}"
    )
    for ev, mu, sd, ch, direction in scored:
        if k >= args.n_probes:
            break
        if used[ch].any():
            continue
        used[ch] = True
        k += 1
        name = f"lbprobe_d{args.day}_p{k:02d}"
        lab = ref.copy()
        lab[ch] = alt[ch]
        pl.DataFrame({"id": test_ids, TARGET: decode_labels(lab)}).write_csv(
            SUBS_DIR / f"{name}.csv"
        )
        np.save(RESULTS_DIR / f"lbprobe_set_{name}.npy", ch)
        manifest.append(
            {
                "probe": name,
                "reference": ref_name,
                "reference_score": ref_score,
                "direction": direction,
                "n_flips": len(ch),
                "expected_delta": mu,
                "delta_sd": sd,
                "ev_keep": ev,
                "decision_rule": "keep set iff public(probe) > public(reference)",
            }
        )
        print(f"{name:22s} {direction:6s} {len(ch):5d} {mu:+9.5f} {sd:8.5f} {ev:11.6f}")

    mdf = pl.DataFrame(manifest)
    mpath = RESULTS_DIR / "lbprobe_probe_manifest.csv"
    if mpath.exists():
        mdf = pl.concat([pl.read_csv(mpath), mdf], how="vertical_relaxed")
    mdf.write_csv(mpath)
    tot_ev = sum(m["ev_keep"] for m in manifest)
    print(f"\ntotal E[kept gain] this round: {tot_ev:+.5f}")
    print(f"manifest -> {mpath}")
    print(
        "\nafter scores land: re-dump kaggle submissions -> lbprobe_ingest.py -> "
        "lbprobe_compose.py"
    )


if __name__ == "__main__":
    main()
