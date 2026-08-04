"""Check a submission against sample_submission before it is ever uploaded.

    python src/verify_submission.py submissions/blend_s1_rank.csv

The row-order invariant is the one failure mode that produces a *valid-looking*
file: right shape, right dtype, plausible values, ~0.5 AUC. So this compares the
id column elementwise against `sample_submission.csv` rather than just checking
the row count, and re-checks the predictions against the `.npy` they came from
when one can be found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def verify(path: Path) -> bool:
    sample = pl.read_csv(DATA_DIR / "sample_submission.csv")
    sub = pl.read_csv(path)
    ok = True

    def check(label: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")

    check("columns", sub.columns == sample.columns, f"{sub.columns}")
    check("row count", sub.height == sample.height, f"{sub.height} vs {sample.height}")
    if sub.height == sample.height:
        same = sub["id"].equals(sample["id"])
        check("ids identical to sample (elementwise)", same, "exact match" if same else "MISALIGNED")
    ids = sub["id"].to_numpy()
    check("ids ascending", bool(np.all(np.diff(ids) > 0)), f"{ids.min()}..{ids.max()}")

    preds = sub[TARGET].to_numpy()
    check("no nulls", not np.isnan(preds).any(), f"{int(np.isnan(preds).sum())} nulls")
    check(
        "in [0, 1]",
        bool(preds.min() >= 0.0 and preds.max() <= 1.0),
        f"min {preds.min():.6f} max {preds.max():.6f} mean {preds.mean():.6f}",
    )
    check("not constant", float(preds.std()) > 1e-9, f"sd {preds.std():.6f}")

    npy = RESULTS_DIR / f"test_{path.stem}.npy"
    if npy.exists():
        arr = np.load(npy).ravel()
        check("matches its .npy", bool(np.allclose(arr, preds, atol=1e-9)), npy.name)
    else:
        print(f"  [skip] no {npy.name} to cross-check against")
    return ok


def main() -> None:
    targets = (
        [Path(a) for a in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(SUBMISSIONS_DIR.glob("*.csv"))
    )
    all_ok = True
    for path in targets:
        print(f"\n{path}")
        all_ok &= verify(path)
    print("\n" + ("ALL SUBMISSIONS OK" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
