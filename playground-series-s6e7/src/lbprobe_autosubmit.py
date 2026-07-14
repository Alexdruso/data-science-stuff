"""LB-probe autonomous round runner (quarantined curiosity track).

Submits a round's probe CSVs (plus optional anchor files), polls until all are
scored, re-dumps the submission list, then runs ingest -> anchor check ->
compose. Quota-aware: submits immediately if today's remaining budget covers
the round, otherwise sleeps to the next UTC reset. Idempotent: skips files
already submitted today, and survives cap errors by sleeping and retrying.

Usage: lbprobe_autosubmit.py --day 2 --extra lbprobe_composed_ref.csv
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "playground-series-s6e7"
COMP_DIR = Path(__file__).resolve().parent.parent
SUBS_DIR = COMP_DIR / "submissions"
RESULTS_DIR = COMP_DIR / "results"
RAW_PATH = RESULTS_DIR / "lbprobe_submissions_raw.csv"

DAILY_CAP = 10
ANCHOR_TOL = 2e-5  # LB 5-dp rounding + accumulated kept-set rounding


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def fetch_all(api: KaggleApi) -> list:
    subs = api.competition_submissions(COMP, page_token=None, page_size=200)
    return list(subs)


def sub_name(s) -> str:
    return getattr(s, "file_name", "") or getattr(s, "fileName", "")


def submitted_today(rows: list) -> list:
    today = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    return [s for s in rows if str(getattr(s, "date", ""))[:10] == today]


def dump_raw(rows: list) -> None:
    with open(RAW_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fileName", "date", "description", "status", "publicScore"])
        for s in rows:
            w.writerow(
                [
                    sub_name(s),
                    getattr(s, "date", ""),
                    getattr(s, "description", ""),
                    str(getattr(s, "status", "")),
                    getattr(s, "public_score", None) or getattr(s, "publicScore", ""),
                ]
            )


def anchor_check(day: int) -> bool:
    """Measured reference score must match the manifest's assumed score."""
    manifest = pl.read_csv(RESULTS_DIR / "lbprobe_probe_manifest.csv").filter(
        pl.col("probe").str.starts_with(f"lbprobe_d{day}_")
    )
    ref_name = manifest["reference"][0]
    assumed = float(manifest["reference_score"][0])
    scores = pl.read_csv(RESULTS_DIR / "lbprobe_scores.csv")
    measured = scores.filter(
        pl.col("reliable") & (pl.col("file_name") == ref_name)
    )["public_score"]
    if measured.is_empty():
        log(f"ANCHOR CHECK FAILED: no reliable score for reference {ref_name}")
        return False
    delta = float(measured[0]) - assumed
    ok = abs(delta) <= ANCHOR_TOL
    log(
        f"anchor check: {ref_name} measured {float(measured[0]):.5f} vs assumed "
        f"{assumed:.5f} (Δ {delta:+.5f}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="extra CSVs (anchors)")
    args = ap.parse_args()

    probe_files = sorted(
        p.name for p in SUBS_DIR.glob(f"lbprobe_d{args.day}_p*.csv")
    )
    round_files = probe_files + list(args.extra)
    assert round_files, f"no lbprobe_d{args.day}_p*.csv files found"
    for fname in round_files:
        assert (SUBS_DIR / fname).exists(), f"missing {fname}"
    messages = {
        f: f"lbprobe round-{args.day} "
        + ("exact-feedback probe" if f in probe_files else "anchor")
        + f" {f}"
        for f in round_files
    }
    log(f"round {args.day}: {len(probe_files)} probes + {len(args.extra)} anchor(s)")

    api = KaggleApi()
    api.authenticate()

    while True:
        rows = fetch_all(api)
        done = {sub_name(s) for s in submitted_today(rows)}
        needed = [f for f in round_files if f not in done]
        remaining = DAILY_CAP - len(submitted_today(rows))
        if not needed or len(needed) <= remaining:
            log(f"quota ok: need {len(needed)}, have {remaining} left today")
            break
        now = datetime.now(timezone.utc)
        reset = (now + timedelta(days=1)).replace(
            hour=0, minute=2, second=0, microsecond=0
        )
        wait = (reset - now).total_seconds()
        log(
            f"quota short (need {len(needed)}, have {remaining}); waiting "
            f"{wait/3600:.2f}h until reset at {reset:%Y-%m-%d %H:%M} UTC"
        )
        time.sleep(max(wait, 0))

    for fname in round_files:
        if fname in done:
            log(f"skip {fname} (already submitted today)")
            continue
        for attempt in range(8):
            try:
                api.competition_submit(str(SUBS_DIR / fname), messages[fname], COMP)
                log(f"submitted {fname}")
                break
            except Exception as e:  # cap not yet reset / transient error
                log(f"{fname} attempt {attempt}: {e}; sleeping 300s")
                time.sleep(300)
        else:
            log(f"GIVING UP on {fname}")
        time.sleep(20)

    log("polling for scores...")
    deadline = time.time() + 45 * 60
    while time.time() < deadline:
        rows = fetch_all(api)
        scored = {sub_name(s): getattr(s, "public_score", None) for s in rows}
        missing = [f for f in round_files if not scored.get(f)]
        if not missing:
            log("all round files scored")
            break
        log(f"waiting on {len(missing)}: {missing[:3]}...")
        time.sleep(120)
    dump_raw(fetch_all(api))
    log(f"raw dump -> {RAW_PATH}")

    py = sys.executable
    src = COMP_DIR / "src"

    def run(script: str) -> bool:
        log(f"running {script}")
        r = subprocess.run(
            [py, str(src / script)], cwd=src, capture_output=True, text=True
        )
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            log(f"{script} FAILED")
            return False
        return True

    if not run("lbprobe_ingest.py"):
        return
    if not anchor_check(args.day):
        log("aborting before compose — fix the additivity model / ingest first")
        return
    if not run("lbprobe_compose.py"):
        return
    # compose overwrote lbprobe_composed_ref.csv; re-run ingest so the stale
    # anchor row is marked unreliable BEFORE any fit/design reads scores.csv
    if not run("lbprobe_ingest.py"):
        return
    log("round complete")


if __name__ == "__main__":
    main()
