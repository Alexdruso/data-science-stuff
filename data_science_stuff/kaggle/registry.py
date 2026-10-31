"""Append-only run registry: one JSON line per run in ``results/runs.jsonl``.

``cv_scores.csv`` records a name and fold scores. That is not enough to
reproduce or audit a run weeks later, so the facts ended up in prose in the
competition ``CLAUDE.md``. The registry stores them as data instead:

* **identity** — ``run_id``, start/finish time, git commit, whether the
  competition/package code was dirty plus a hash of that diff;
* **inputs** — the exact command line, ``params``, ``config_hash`` over
  params + command + inputs (lesson 12: a key must encode every dimension
  that varies), ``inputs`` (runs this one consumed — the lineage of a stack);
* **outputs** — OOF score, per-fold scores, artifact paths;
* **protocol** — the pre-registration id and whether its ``prereg:`` commit
  predates the run (see :mod:`data_science_stuff.kaggle.prereg`), the verdict.

The file is append-only. Later facts (a verdict decided after the run) are
added with :func:`annotate_run` as separate ``annotate`` lines and folded in
by :func:`read_runs`, so a crash mid-write can only ever lose one line.
``runs.jsonl`` is small text: commit it with the competition.

CLI::

    python -m data_science_stuff.kaggle runs list results/
    python -m data_science_stuff.kaggle runs import-cv-scores results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_science_stuff.kaggle._git import git
from data_science_stuff.kaggle.prereg import find_prereg

RUNS_FILE = "runs.jsonl"

# Default ``started_at``: when this module was first imported, i.e. near the
# start of the training script. Pass ``started_at`` explicitly to be exact.
PROCESS_START = datetime.now(timezone.utc)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GitState:
    """Code version of a run. ``None`` fields mean "not in a git repository"."""

    sha: str | None
    dirty: bool | None
    dirty_paths: list[str]
    diff_hash: str | None


@dataclass
class RunRecord:
    """One registry entry. Serialised as a JSON line with ``event = "run"``."""

    run_id: str
    name: str
    started_at: str
    finished_at: str
    command: list[str]
    config_hash: str
    git: GitState
    metric_name: str = "score"
    oof_score: float | None = None
    fold_scores: list[float] = field(default_factory=list)
    n_folds: int | None = None
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    prereg: str | None = None
    prereg_sha: str | None = None
    prereg_committed_at: str | None = None
    prereg_before_run: bool | None = None
    verdict: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _json_default(obj: object) -> object:
    """Make numpy scalars/arrays and paths JSON-serialisable."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(obj)


def _clean_float(x: float | None) -> float | None:
    if x is None:
        return None
    value = float(x)
    return None if math.isnan(value) else value


def _to_json(obj: object) -> Any:
    """Round-trip through JSON so stored params are exactly what reads back."""
    return json.loads(json.dumps(obj, default=_json_default))


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable 12-hex-digit hash of a JSON-serialisable mapping (key order ignored).

    >>> config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    True
    >>> len(config_hash({}))
    12
    """
    blob = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_state(paths: Sequence[Path | str]) -> GitState:
    """Commit and dirtiness of ``paths`` (tracked files only).

    Only the given paths are checked, so unrelated uncommitted work elsewhere
    in a monorepo does not mark every run dirty. ``diff_hash`` fingerprints
    the uncommitted diff, so two dirty runs with the same hash ran the same code.
    """
    resolved = [Path(p).resolve() for p in paths]
    cwd = resolved[0] if resolved[0].is_dir() else resolved[0].parent
    top = git("rev-parse", "--show-toplevel", cwd=cwd)
    sha = git("rev-parse", "HEAD", cwd=cwd)
    if top is None or sha is None:
        return GitState(None, None, [], None)
    root = Path(top).resolve()
    specs = [str(p) for p in resolved if p == root or root in p.parents]
    # Tracked files changed vs HEAD (staged or not); untracked files are ignored.
    changed = git("diff", "--name-only", "HEAD", "--", *specs, cwd=cwd)
    dirty_paths = [line for line in (changed or "").splitlines() if line]
    diff_hash = None
    if dirty_paths:
        diff = git("diff", "HEAD", "--", *specs, cwd=cwd) or ""
        diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:12]
    return GitState(sha, bool(dirty_paths), dirty_paths, diff_hash)


def _append(results_dir: Path, payload: Mapping[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, default=_json_default, allow_nan=False)
    with (results_dir / RUNS_FILE).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_run(
    results_dir: Path,
    name: str,
    *,
    oof_score: float | None = None,
    fold_scores: Sequence[float] = (),
    metric_name: str = "score",
    params: Mapping[str, Any] | None = None,
    inputs: Sequence[str] = (),
    artifacts: Sequence[Path | str] = (),
    prereg: str | None = None,
    verdict: str | None = None,
    notes: str | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
    started_at: datetime | None = None,
    command: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RunRecord:
    """Append one run to ``results_dir/runs.jsonl`` and return its record.

    Args:
        results_dir: The competition's ``results/`` directory.
        name: Model/run name, e.g. ``"lgbm_route_v2"`` (the ``oof_<name>``
            artifact stem). Several runs may share a name; ``run_id`` is unique.
        oof_score: Overall out-of-fold metric.
        fold_scores: Per-fold metric values.
        metric_name: e.g. ``"auc"``.
        params: Hyperparameters / switches. Stored and hashed.
        inputs: Names or run ids this run consumed (stack members, teachers).
        artifacts: Files written (prediction parquet, submission CSV).
        prereg: Pre-registration id (``"PR-59"``). Its ``prereg:`` commit is
            looked up and ``prereg_before_run`` records whether it came first.
        verdict: e.g. ``"PASS"`` / ``"FAIL"``; can be added later with
            :func:`annotate_run`.
        notes: Free text.
        n_folds: Defaults to ``len(fold_scores)`` when scores are given.
        seed: CV / model seed.
        started_at: Run start; defaults to when this module was imported.
        command: Defaults to ``sys.argv``.
        extra: Anything else worth keeping (device, runtime, ...).
    """
    finished = datetime.now(timezone.utc)
    started = started_at or PROCESS_START
    if started.tzinfo is None:
        started = started.astimezone()
    cmd = list(command) if command is not None else list(sys.argv)
    clean_params = _to_json(dict(params or {}))
    input_list = [str(x) for x in inputs]
    competition_root = results_dir.resolve().parent
    state = git_state([competition_root, _PACKAGE_DIR])

    prereg_sha = prereg_at = None
    before: bool | None = None
    if prereg is not None:
        found = find_prereg(
            prereg, cwd=competition_root if competition_root.exists() else None
        )
        if found is None:
            warnings.warn(
                f"no 'prereg: {prereg}' commit found: the run is logged as NOT pre-registered",
                stacklevel=2,
            )
            before = False
        else:
            prereg_sha, prereg_at = found.sha, found.committed_at.isoformat()
            before = found.committed_at <= started

    scores = [float(s) for s in fold_scores]
    record = RunRecord(
        run_id=f"{name}-{finished:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
        name=name,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        command=cmd,
        config_hash=config_hash(
            {"params": clean_params, "command": cmd, "inputs": input_list}
        ),
        git=state,
        metric_name=metric_name,
        oof_score=_clean_float(oof_score),
        fold_scores=scores,
        n_folds=n_folds if n_folds is not None else (len(scores) or None),
        seed=seed,
        params=clean_params,
        inputs=input_list,
        artifacts=[str(a) for a in artifacts],
        prereg=prereg,
        prereg_sha=prereg_sha,
        prereg_committed_at=prereg_at,
        prereg_before_run=before,
        verdict=verdict,
        notes=notes,
        extra=_to_json(dict(extra or {})),
    )
    _append(results_dir, {"event": "run", **asdict(record)})
    return record


def annotate_run(results_dir: Path, run_id: str, **fields: Any) -> None:
    """Record facts decided after a run (``verdict="FAIL"``, ``notes=...``).

    Appends an ``annotate`` line; :func:`read_runs` applies it on top of the
    run, later annotations winning. The original run line is never rewritten.
    """
    known = {r["run_id"] for r in read_runs(results_dir)}
    if run_id not in known:
        msg = f"unknown run_id {run_id!r} in {results_dir / RUNS_FILE}"
        raise KeyError(msg)
    _append(
        results_dir,
        {
            "event": "annotate",
            "run_id": run_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "fields": fields,
        },
    )


def read_runs(results_dir: Path) -> list[dict[str, Any]]:
    """All runs in file order, with annotations applied.

    A truncated or corrupt line (e.g. the machine died mid-write) is skipped
    with a warning rather than making the whole history unreadable.
    """
    path = results_dir / RUNS_FILE
    if not path.exists():
        return []
    runs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(f"{path}:{lineno}: unreadable line skipped", stacklevel=2)
                continue
            event = entry.pop("event", "run")
            if event == "run":
                runs[entry["run_id"]] = entry
            elif event == "annotate" and entry.get("run_id") in runs:
                runs[entry["run_id"]].update(entry.get("fields", {}))
    return list(runs.values())


def load_runs(results_dir: Path) -> pd.DataFrame:
    """:func:`read_runs` as a DataFrame (``git`` flattened to ``git_*`` columns)."""
    rows = []
    for run in read_runs(results_dir):
        row = dict(run)
        for key, value in (row.pop("git", None) or {}).items():
            row[f"git_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def import_cv_scores(results_dir: Path) -> int:
    """Backfill the registry from a legacy ``cv_scores.csv``.

    Each CSV row becomes a run with ``extra.source = "cv_scores.csv"`` and
    whatever the CSV knew (name, time, scores). Rows already imported are
    skipped, so this is safe to re-run. Returns the number of rows added.
    """
    csv_path = results_dir / "cv_scores.csv"
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    oof_cols = [c for c in df.columns if c.startswith("oof_")]
    if not oof_cols or "model" not in df.columns:
        msg = f"{csv_path} has no model/oof_* columns"
        raise ValueError(msg)
    seen = {
        r.get("extra", {}).get("legacy_key")
        for r in read_runs(results_dir)
        if r.get("extra", {}).get("source") == "cv_scores.csv"
    }
    added = 0
    for _, row in df.iterrows():
        oof_col = next((c for c in oof_cols if pd.notna(row[c])), oof_cols[0])
        metric = oof_col[len("oof_") :]
        stamp = str(row.get("timestamp", ""))
        key = f"{row['model']}|{stamp}"
        if key in seen:
            continue
        fold_cols = [
            c for c in df.columns if c.startswith("fold_") and c.endswith(f"_{metric}")
        ]
        folds = [float(row[c]) for c in fold_cols if pd.notna(row[c])]
        when = pd.to_datetime(stamp, utc=True, errors="coerce")
        iso = when.isoformat() if pd.notna(when) else ""
        record = RunRecord(
            run_id=f"{row['model']}-legacy-{config_hash({'k': key})[:8]}",
            name=str(row["model"]),
            started_at=iso,
            finished_at=iso,
            command=[],
            config_hash="",
            git=GitState(None, None, [], None),
            metric_name=metric,
            oof_score=_clean_float(row[oof_col]),
            fold_scores=folds,
            n_folds=len(folds) or None,
            extra={"source": "cv_scores.csv", "legacy_key": key},
        )
        _append(results_dir, {"event": "run", **asdict(record)})
        seen.add(key)
        added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    """Sub-CLI behind ``python -m data_science_stuff.kaggle``."""
    parser = argparse.ArgumentParser(
        description="Inspect or backfill results/runs.jsonl"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="print the runs as a table")
    p_list.add_argument("results_dir", type=Path)
    p_list.add_argument("-n", type=int, default=30, help="last N runs (0 = all)")
    p_imp = sub.add_parser("import-cv-scores", help="backfill from cv_scores.csv")
    p_imp.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "import-cv-scores":
        print(f"imported {import_cv_scores(args.results_dir)} rows")
        return 0
    df = load_runs(args.results_dir)
    if df.empty:
        print("no runs")
        return 0
    cols = [
        c
        for c in (
            "run_id",
            "oof_score",
            "prereg",
            "prereg_before_run",
            "verdict",
            "git_dirty",
        )
        if c in df
    ]
    view = df[cols] if args.n == 0 else df[cols].tail(args.n)
    print(view.to_string(index=False))
    return 0
