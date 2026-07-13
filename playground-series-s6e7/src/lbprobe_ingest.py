"""LB-probe step 1 — reconcile Kaggle submission scores with disk CSVs.

QUARANTINE NOTE: this file and everything named lbprobe_* is the explicitly
user-sanctioned public-LB curiosity track (2026-07-12). Scores here must never
feed finals selection, CLAUDE.md experiment logs, or memory. Finals stay
CV-selected per the standing LB-blind protocol.

Input : results/lbprobe_submissions_raw.csv (dump of `kaggle competitions
        submissions`, timestamps UTC).
Output: results/lbprobe_scores.csv — one row per (fileName, submission), with a
        `reliable` flag: the disk file's bytes are what was submitted iff the
        submission happened AFTER the file's last modification (mtime rule).
        Multiple reliable submissions of one file must agree on score.
"""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

COMP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = COMP_DIR / "results"
SUBS_DIR = COMP_DIR / "submissions"

RAW_PATH = RESULTS_DIR / "lbprobe_submissions_raw.csv"
OUT_PATH = RESULTS_DIR / "lbprobe_scores.csv"


def load_raw() -> pl.DataFrame:
    df = pl.read_csv(RAW_PATH, schema_overrides={"publicScore": pl.Float64})
    df = df.filter(
        pl.col("status").str.contains("COMPLETE") & pl.col("publicScore").is_not_null()
    )
    return df.with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S%.f").alias("submitted_utc")
    )


def main() -> None:
    raw = load_raw()
    rows: list[dict[str, object]] = []
    for rec in raw.iter_rows(named=True):
        fname = rec["fileName"]
        disk = SUBS_DIR / fname
        sub_time = rec["submitted_utc"].replace(tzinfo=timezone.utc)
        if not disk.exists():
            reliable, reason, mtime_utc = False, "no disk file", None
        else:
            mtime_utc = datetime.fromtimestamp(disk.stat().st_mtime, tz=timezone.utc)
            if sub_time >= mtime_utc:
                reliable, reason = True, ""
            else:
                reliable, reason = False, "file rebuilt after this submission"
        rows.append(
            {
                "file_name": fname,
                "submitted_utc": sub_time.replace(tzinfo=None),
                "public_score": rec["publicScore"],
                "mtime_utc": mtime_utc.replace(tzinfo=None) if mtime_utc else None,
                "reliable": reliable,
                "quarantine_reason": reason,
                "description": rec["description"] or "",
            }
        )
    df = pl.DataFrame(rows).sort("submitted_utc")

    # Consistency: reliable submissions of the same file share the same bytes,
    # so they must share the same score. Violations poison the whole file.
    bad_files = (
        df.filter(pl.col("reliable"))
        .group_by("file_name")
        .agg(pl.col("public_score").n_unique().alias("n_scores"))
        .filter(pl.col("n_scores") > 1)["file_name"]
        .to_list()
    )
    if bad_files:
        df = df.with_columns(
            pl.when(pl.col("file_name").is_in(bad_files))
            .then(pl.lit(False))
            .otherwise(pl.col("reliable"))
            .alias("reliable"),
            pl.when(pl.col("file_name").is_in(bad_files))
            .then(pl.lit("inconsistent scores among reliable subs"))
            .otherwise(pl.col("quarantine_reason"))
            .alias("quarantine_reason"),
        )

    df.write_csv(OUT_PATH)

    usable = (
        df.filter(pl.col("reliable"))
        .group_by("file_name")
        .agg(pl.col("public_score").first())
        .sort("public_score", descending=True)
    )
    print(f"total scored submissions : {len(df)}")
    print(f"reliable (file==bytes)   : {df['reliable'].sum()}")
    print(f"distinct usable files    : {len(usable)}")
    print(f"inconsistent files       : {bad_files}")
    print("\nquarantined:")
    for r in df.filter(~pl.col("reliable")).iter_rows(named=True):
        print(
            f"  {r['file_name']:32s} {r['public_score']:.5f} "
            f"sub={r['submitted_utc']} — {r['quarantine_reason']}"
        )
    print("\nusable (file → public score), best first:")
    for r in usable.iter_rows(named=True):
        print(f"  {r['file_name']:32s} {r['public_score']:.5f}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
