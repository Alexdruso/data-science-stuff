"""Day-13 EVE — label-channel split-consistency audit (forensics idea #2).

Every saturation proof is train-internal; the masks are PROVEN split-
inconsistent (the campaign's biggest find). If any label-affecting generator
step also differed per split, train-fitted cell posteriors are miscalibrated
on test. The quarantined lbprobe track holds the only test-label-touching
data ever collected: 27 flip sets with EXACT public deltas. This compares
each measured delta to its prediction under the TRAIN-ONLY prior posterior
(calibrated_prior — no public-score conditioning, bypassing lbprobe_fit's
circularity), grouped by round and flip direction. Group-mean residuals past
the sampling noise = split-inconsistent label channel (a lead, exploitation
user-gated); flat = the train posterior is calibrated on real test labels =
the saturation proofs extend across the split.

Quarantine-safe output: deltas and residuals only, no absolute scores.

Run: python src/lbprobe_split_audit.py -> results/lbprobe_split_audit.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from lbprobe_common import calibrated_prior, load_submission_matrix, load_test_ids
from train_common import RESULTS_DIR

SUBS_DIR = Path(__file__).parent.parent / "submissions"


def main() -> None:
    scores = pl.read_csv(RESULTS_DIR / "lbprobe_scores.csv").filter(
        pl.col("reliable")
    )
    sc = dict(zip(scores["file_name"], scores["public_score"]))
    man = pl.read_csv(RESULTS_DIR / "lbprobe_probe_manifest.csv")

    q = calibrated_prior(quiet=True)  # (n_test, 3) train-only posterior
    n_prior = q.sum(axis=0)  # expected test class counts
    test_ids = load_test_ids()

    # v[i, c] = expected public-bacc contribution of assigning class c to row i
    v = q / (3.0 * n_prior)

    # reconstruct each round's reference labels by replaying compositions
    ref_labels: dict[str, np.ndarray] = {}
    first_ref = man["reference"][0]
    ref_labels[first_ref] = load_submission_matrix([first_ref], test_ids)[:, 0]

    lines = ["lbprobe split-consistency audit (train-prior predictions, deltas only)"]
    rows_out = []
    current_ref = ref_labels[first_ref].copy()
    prev_round_ref = first_ref
    for ref_name in man["reference"].unique(maintain_order=True):
        batch = man.filter(pl.col("reference") == ref_name)
        if ref_name != prev_round_ref:
            prev_round_ref = ref_name
        kept_sets = []
        for r in batch.iter_rows(named=True):
            fname = r["probe"] + ".csv"
            if fname not in sc or r["reference_score"] is None:
                continue
            ch = np.load(RESULTS_DIR / f"lbprobe_set_{r['probe']}.npy")
            c_from, c_to = (int(x) for x in r["direction"].split("->"))
            # sanity: reference labels at those rows should be c_from
            frac_from = float((current_ref[ch] == c_from).mean())
            pred = float((v[ch, c_to] - v[ch, c_from]).sum())
            meas = float(sc[fname] - r["reference_score"])
            resid = meas - pred
            rows_out.append(
                dict(probe=r["probe"], direction=r["direction"], n=len(ch),
                     pred=pred, meas=meas, resid=resid, sd=float(r["delta_sd"]),
                     ref_ok=frac_from)
            )
            if meas > 0:
                kept_sets.append((ch, c_to))
        # replay composition for the next round's reference
        for ch, c_to in kept_sets:
            current_ref[ch] = c_to

    lines.append(f"{'probe':<18} {'dir':<6} {'n':>4} {'pred':>9} {'meas':>9} "
                 f"{'resid':>9} {'sd':>8} {'refok':>6}")
    for r in rows_out:
        lines.append(
            f"{r['probe']:<18} {r['direction']:<6} {r['n']:>4} {r['pred']:>+9.5f} "
            f"{r['meas']:>+9.5f} {r['resid']:>+9.5f} {r['sd']:>8.5f} {r['ref_ok']:>6.2f}"
        )

    # grouped residual tests
    import collections
    groups = collections.defaultdict(list)
    for r in rows_out:
        groups[("dir", r["direction"])].append(r)
        groups[("round", r["probe"].split("_")[1][:2])].append(r)
    groups[("all", "all")] = rows_out
    lines.append("\ngroup residuals (mean ± noise-implied se; |z|>2.5 = signal):")
    for (g, k), rs in sorted(groups.items()):
        res = np.array([r["resid"] for r in rs])
        se = np.sqrt(np.sum([r["sd"] ** 2 for r in rs])) / len(rs)
        z = res.mean() / max(se, 1e-9)
        lines.append(
            f"  {g}={k:<6} n={len(rs):>2}  mean_resid={res.mean():+.6f} "
            f"se={se:.6f}  z={z:+.2f}"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "lbprobe_split_audit.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
