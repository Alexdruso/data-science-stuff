"""Verifiable pre-registration: commit the bar before the number exists.

Campaign lesson 10 says to write the arm, instrument, prior and pass/fail bar
into the competition ``CLAUDE.md`` *before* running. Writing it down is not
enough to prove the order: if the pre-registration and its result land in the
same commit, nobody (including you, a week later) can tell which came first.

:func:`commit_prereg` commits only the ``CLAUDE.md`` with the subject
``prereg: <PR-id>``. The run registry
(:func:`data_science_stuff.kaggle.registry.log_run` with ``prereg=``) then
looks that commit up and records whether it predates the run's start.

CLI::

    python -m data_science_stuff.kaggle prereg PR-59 playground-series-s7e1/CLAUDE.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data_science_stuff.kaggle._git import git

PREREG_PREFIX = "prereg: "
_LOG_FIELDS = 3  # sha, commit date, subject
_PR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


@dataclass(frozen=True)
class PreregCommit:
    """The commit that registered a pre-registration."""

    pr_id: str
    sha: str
    committed_at: datetime


def _check_pr_id(pr_id: str) -> None:
    if not _PR_ID.match(pr_id):
        msg = f"invalid pre-registration id {pr_id!r} (use e.g. 'PR-59', no spaces)"
        raise ValueError(msg)


def find_prereg(pr_id: str, cwd: Path | str | None = None) -> PreregCommit | None:
    """Return the *earliest* ``prereg: <pr_id>`` commit, or ``None``.

    The earliest one counts because the bar must not move after the number
    lands: a later ``prereg:`` commit for the same id is an amendment, and the
    registry should be judged against the original.

    Matching is on the exact subject, so ``PR-5`` never matches ``PR-55``.
    """
    _check_pr_id(pr_id)
    subject = f"{PREREG_PREFIX}{pr_id}"
    out = git(
        "log",
        "--format=%H%x09%cI%x09%s",
        "--fixed-strings",
        f"--grep={subject}",
        cwd=cwd,
    )
    if not out:
        return None
    hits = [
        parts
        for parts in (line.split("\t", 2) for line in out.splitlines())
        if len(parts) == _LOG_FIELDS and parts[-1] == subject
    ]
    if not hits:
        return None
    sha, when, _ = hits[-1]  # git log is newest-first
    return PreregCommit(pr_id, sha, datetime.fromisoformat(when))


def commit_prereg(
    pr_id: str, claude_md: Path | str, *, message: str | None = None
) -> PreregCommit:
    """Commit ``claude_md`` (and only it) as the pre-registration ``pr_id``.

    Args:
        pr_id: Pre-registration id, e.g. ``"PR-59"``. Must appear in the
            file's uncommitted changes, which guards against committing the
            wrong edit.
        claude_md: The competition ``CLAUDE.md`` holding the registration.
        message: Optional commit body (e.g. the bar, for ``git log`` readers).

    Returns:
        The new :class:`PreregCommit`.

    Raises:
        ValueError: the id is malformed, the file has no uncommitted changes,
            or the changes do not mention ``pr_id``.
        RuntimeError: git failed (not a repository, commit hook refused, ...).
    """
    _check_pr_id(pr_id)
    path = Path(claude_md).resolve()
    cwd = path.parent
    diff = git("diff", "HEAD", "--", str(path), cwd=cwd)
    if diff is None:
        msg = f"git diff failed for {path} (not in a git repository?)"
        raise RuntimeError(msg)
    if not diff:
        msg = (
            f"{path.name} has no uncommitted changes: write the pre-registration first"
        )
        raise ValueError(msg)
    added = "\n".join(
        line
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if pr_id not in added:
        msg = f"the uncommitted changes in {path.name} do not mention {pr_id}"
        raise ValueError(msg)

    args = ["commit", "-m", f"{PREREG_PREFIX}{pr_id}"]
    if message:
        args += ["-m", message]
    # A pathspec makes this an --only commit: other staged work stays staged.
    if git(*args, "--", str(path), cwd=cwd) is None:
        msg = f"git commit failed for {path}"
        raise RuntimeError(msg)
    sha = git("rev-parse", "HEAD", cwd=cwd)
    when = git("show", "-s", "--format=%cI", "HEAD", cwd=cwd)
    if sha is None or when is None:  # pragma: no cover - HEAD exists after a commit
        msg = "could not read back the pre-registration commit"
        raise RuntimeError(msg)
    return PreregCommit(pr_id, sha, datetime.fromisoformat(when))


def main(argv: list[str] | None = None) -> int:
    """Sub-CLI behind ``python -m data_science_stuff.kaggle``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pr_id", help="pre-registration id, e.g. PR-59")
    parser.add_argument("claude_md", type=Path, help="competition CLAUDE.md")
    parser.add_argument("-m", "--message", help="optional commit body (the bar)")
    args = parser.parse_args(argv)
    found = commit_prereg(args.pr_id, args.claude_md, message=args.message)
    print(
        f"registered {found.pr_id} at {found.committed_at.isoformat()} ({found.sha[:10]})"
    )
    print(
        f"log the run with prereg={found.pr_id!r} so the registry can verify the order"
    )
    return 0
