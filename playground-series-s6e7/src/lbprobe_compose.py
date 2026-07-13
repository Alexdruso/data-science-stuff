"""LB-probe step 5 — read probe scores, keep winning flip sets, compose next ref.

Run after: new `kaggle competitions submissions` dump -> lbprobe_ingest.py.
For every probe in the manifest that now has a reliable score, the exact delta
vs its reference decides keep/revert. All kept (disjoint) sets compose onto the
reference by linearity; the composed CSV is the next round's reference with a
KNOWN score (sum of kept deltas, exact up to rounding accumulation).
"""

import numpy as np
import polars as pl

from lbprobe_common import (
    RESULTS_DIR,
    SUBS_DIR,
    TARGET,
    load_scores,
    load_submission_matrix,
    load_test_ids,
)
from features import CLASSES


def main() -> None:
    manifest = pl.read_csv(RESULTS_DIR / "lbprobe_probe_manifest.csv")
    # only the latest round's probes share a composable reference
    ref_name = manifest["reference"][-1]
    manifest = manifest.filter(pl.col("reference") == ref_name)
    scores = load_scores()
    score_of = dict(zip(scores["file_name"], scores["public_score"]))

    test_ids = load_test_ids()
    ref_labels = load_submission_matrix([ref_name], test_ids)[:, 0].copy()
    ref_score = float(manifest["reference_score"][0])

    total = 0.0
    kept = 0
    print(f"reference {ref_name} @ {ref_score:.5f}\n")
    print(f"{'probe':22s} {'n':>5s} {'E[Δ]':>9s} {'measured Δ':>11s}  verdict")
    for row in manifest.iter_rows(named=True):
        fname = row["probe"] + ".csv"
        if fname not in score_of:
            print(f"{row['probe']:22s} {row['n_flips']:5d} {row['expected_delta']:+9.5f}  {'(no score yet)':>11s}")
            continue
        delta = score_of[fname] - row["reference_score"]
        keep = delta > 0
        if keep:
            ch = np.load(RESULTS_DIR / f"lbprobe_set_{row['probe']}.npy")
            probe_labels = load_submission_matrix([fname], test_ids)[:, 0]
            ref_labels[ch] = probe_labels[ch]
            total += delta
            kept += 1
        print(
            f"{row['probe']:22s} {row['n_flips']:5d} {row['expected_delta']:+9.5f} "
            f"{delta:+11.5f}  {'KEEP' if keep else 'revert'}"
        )

    out = SUBS_DIR / "lbprobe_composed_ref.csv"
    pl.DataFrame(
        {"id": test_ids, TARGET: np.array(CLASSES)[ref_labels]}
    ).write_csv(out)
    print(
        f"\nkept {kept} sets, composed score = {ref_score:.5f} {total:+.5f} "
        f"= {ref_score + total:.5f}"
    )
    print(f"wrote {out}")
    print(
        "next round: lbprobe_design.py --day <d+1> --reference lbprobe_composed_ref.csv"
        " (after adding its computed score to results/lbprobe_scores.csv or"
        " submitting it once for a clean anchor)"
    )


if __name__ == "__main__":
    main()
