"""Minimal git helpers shared by the run registry and the pre-registration CLI."""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv lists, never a shell
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
