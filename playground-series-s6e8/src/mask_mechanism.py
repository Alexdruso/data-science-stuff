"""Closes the pre-registered day-1 experiment on the missingness mechanism.

Arm: does the train<->test null-rate discrepancy reflect a different
missingness *mechanism*, or just different marginal rates on the same rule?

Instruments (named in CLAUDE.md BEFORE the numbers landed — do not substitute):
  * `shift.mask_mechanism_report` — per-column MAR AUC, compared across splits
  * `shift.unseen_mask_patterns`  — test rows whose pattern is impossible in train
  * `shift.adversarial_validation` on the 12-column mask matrix alone

Bar (does not move): a per-column MAR AUC gap >= 0.02 on >= 2 columns, OR
adversarial AUC on the mask matrix >= 0.60. Below that the mechanism is
declared the same and the avenue closes for mask-*repair* work. It does not
close mask *features*, and it does not close mask x value interactions.

Prior stated before running: null.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.shift import (
    adversarial_validation,
    mask_frame,
    mask_mechanism_report,
    unseen_mask_patterns,
)

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

GAP_BAR = 0.02
GAP_MIN_COLUMNS = 2
ADV_BAR = 0.60


def main() -> None:
    # Through build_features() so the row-order invariant holds even here.
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))

    raw_cols = [c for c in test.columns if c not in EXCLUDE_COLS]
    # Only the 12 *raw* columns carry missingness of their own; the derived
    # ratios inherit NaN from their operands and would double-count it.
    raw_cols = [c for c in raw_cols if c in set(pl.read_csv(DATA_DIR / "test.csv").columns)]
    train_pd = train.select(raw_cols).to_pandas()
    test_pd = test.select(raw_cols).to_pandas()
    print(f"columns under test ({len(raw_cols)}): {raw_cols}\n")

    print("=== instrument 1: per-column MAR AUC (mask ~ other masks) ===")
    report = mask_mechanism_report(train_pd, test_pd)
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    over = report[report["auc_gap"] >= GAP_BAR]
    print(f"\ncolumns with auc_gap >= {GAP_BAR}: {len(over)} (bar: >= {GAP_MIN_COLUMNS})")

    print("\n=== instrument 2: adversarial validation on the MASK matrix alone ===")
    adv = adversarial_validation(mask_frame(train_pd), mask_frame(test_pd))
    print(f"mask-matrix adversarial AUC: {adv.auc:.4f} (bar: >= {ADV_BAR})")

    print("\n=== instrument 3: unseen mask patterns ===")
    unseen = unseen_mask_patterns(train_pd, test_pd)
    print(
        f"test rows with a train-impossible pattern: {int(unseen.sum())} "
        f"({unseen.mean() * 100:.4f}%)"
    )

    passed = (len(over) >= GAP_MIN_COLUMNS) or (adv.auc >= ADV_BAR)
    print("\n" + "=" * 72)
    print(f"VERDICT vs the unmoved bar: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Mechanism differs -> mask-repair work is on the table (surface change!).")
    else:
        print("Mechanism declared the SAME. Mask-repair avenue closes.")
        print("Not closed: mask features, and mask x value interactions.")
    print("=" * 72)

    out = RESULTS_DIR / "mask_mechanism_report.csv"
    report.to_csv(out, index=False)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
