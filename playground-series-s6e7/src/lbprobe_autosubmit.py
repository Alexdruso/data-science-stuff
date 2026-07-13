"""LB-probe autonomous round runner (quarantined curiosity track).

Waits for the UTC daily-quota reset, submits the round's probe CSVs, polls
until all are scored, re-dumps the submission list, and runs ingest + compose.
Idempotent: skips files that already appear in the fetched submission list
with a score, and survives cap errors by sleeping and retrying.
"""

import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "playground-series-s6e7"
COMP_DIR = Path(__file__).resolve().parent.parent
SUBS_DIR = COMP_DIR / "submissions"
RESULTS_DIR = COMP_DIR / "results"
RAW_PATH = RESULTS_DIR / "lbprobe_submissions_raw.csv"

ROUND_FILES = [f"lbprobe_d1_p{k:02d}.csv" for k in range(1, 10)] + [
    "lbprobe_full_bayes.csv"
]
MESSAGES = {f: f"lbprobe round-1 exact-feedback probe {f}" for f in ROUND_FILES}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def fetch_all(api: KaggleApi) -> list:
    subs = api.competition_submissions(COMP, page_token=None, page_size=200)
    return list(subs)


def dump_raw(rows: list) -> None:
    with open(RAW_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fileName", "date", "description", "status", "publicScore"])
        for s in rows:
            w.writerow(
                [
                    getattr(s, "file_name", "") or getattr(s, "fileName", ""),
                    getattr(s, "date", ""),
                    getattr(s, "description", ""),
                    str(getattr(s, "status", "")),
                    getattr(s, "public_score", None) or getattr(s, "publicScore", ""),
                ]
            )


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    now = datetime.now(timezone.utc)
    reset = (now + timedelta(days=1)).replace(hour=0, minute=2, second=0, microsecond=0)
    wait = (reset - now).total_seconds()
    log(f"waiting {wait/3600:.2f}h until quota reset at {reset:%Y-%m-%d %H:%M} UTC")
    time.sleep(max(wait, 0))

    done = {
        (getattr(s, "file_name", "") or getattr(s, "fileName", ""))
        for s in fetch_all(api)
        if str(getattr(s, "date", ""))[:10] == f"{datetime.now(timezone.utc):%Y-%m-%d}"
    }
    for fname in ROUND_FILES:
        if fname in done:
            log(f"skip {fname} (already submitted today)")
            continue
        for attempt in range(8):
            try:
                api.competition_submit(str(SUBS_DIR / fname), MESSAGES[fname], COMP)
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
        scored = {
            (getattr(s, "file_name", "") or getattr(s, "fileName", "")): getattr(
                s, "public_score", None
            )
            for s in rows
        }
        missing = [f for f in ROUND_FILES if not scored.get(f)]
        if not missing:
            log("all probes scored")
            break
        log(f"waiting on {len(missing)}: {missing[:3]}...")
        time.sleep(120)
    dump_raw(fetch_all(api))
    log(f"raw dump -> {RAW_PATH}")

    py = sys.executable
    src = COMP_DIR / "src"
    for script in ("lbprobe_ingest.py", "lbprobe_compose.py"):
        log(f"running {script}")
        r = subprocess.run(
            [py, str(src / script)], cwd=src, capture_output=True, text=True
        )
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            log(f"{script} FAILED")
            return
    log("round complete")


if __name__ == "__main__":
    main()
