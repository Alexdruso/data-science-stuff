"""Command line for the run registry: ``python -m data_science_stuff.kaggle <command>``.

Commands::

    runs list <results_dir> [-n N]        show the registry as a table
    runs import-cv-scores <results_dir>   backfill from a legacy cv_scores.csv
    prereg <PR-id> <CLAUDE.md> [-m BAR]   commit a pre-registration (git)
    mlflow-sync <results_dir>...          mirror runs.jsonl into mlflow.db
"""

from __future__ import annotations

import sys

from data_science_stuff.kaggle import prereg, registry

USAGE = __doc__.split("Commands::", 1)[1]


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the registry, pre-registration or MLflow sub-CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = (args[0], args[1:]) if args else ("", [])
    if command == "runs":
        return registry.main(rest)
    if command == "prereg":
        return prereg.main(rest)
    if command == "mlflow-sync":
        from data_science_stuff.kaggle import tracking  # noqa: PLC0415 - optional extra

        return tracking.main(rest)
    print(f"usage: python -m data_science_stuff.kaggle <command>\n{USAGE}")
    return 0 if command in {"", "-h", "--help"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
