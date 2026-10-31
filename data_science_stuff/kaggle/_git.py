"""Minimal git helpers shared by the run registry and the pre-registration CLI."""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv lists, never a shell
from datetime import datetime
from pathlib import Path


def git(*args: str, cwd: Path | str | None = None) -> str | None:
    """Run ``git <args>`` and return its stripped stdout.

    Returns ``None`` when git is missing, ``cwd`` is not inside a repository,
    or the command fails, so callers can record "unknown" instead of crashing
    a training run over bookkeeping.
    """
    # Fixed argv list, no shell; "git" is resolved on PATH like any CLI user's.
    try:
        done = subprocess.run(  # noqa: S603  # nosec B603, B607
            ["git", *args],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def parse_iso(stamp: str) -> datetime:
    """``datetime.fromisoformat`` that also accepts a trailing ``Z`` (UTC).

    Git prints UTC commit times as ``...Z`` (``--format=%cI``), which
    ``fromisoformat`` only understands from Python 3.11 on.

    >>> parse_iso("2026-10-31T16:42:17Z").utcoffset().total_seconds()
    0.0
    """
    if stamp.endswith(("Z", "z")):
        stamp = stamp[:-1] + "+00:00"
    return datetime.fromisoformat(stamp)
