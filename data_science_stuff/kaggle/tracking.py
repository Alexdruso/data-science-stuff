"""Mirror ``results/runs.jsonl`` into a local MLflow store for browsing.

The JSONL registry (:mod:`data_science_stuff.kaggle.registry`) stays the
source of truth: it is plain text, committed with the competition, and needs
no server. MLflow is a *view* on it — a UI to sort, filter and chart runs.
Syncing is idempotent: each MLflow run carries the registry ``run_id`` as a
tag, existing runs only get their tags refreshed (so a verdict added later
with ``annotate_run`` shows up), and nothing is uploaded except metadata.

Needs the optional extra: ``uv pip install -e ".[tracking]"``.

CLI::

    python -m data_science_stuff.kaggle mlflow-sync playground-series-*/results
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # or: make mlflow-ui
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from data_science_stuff.kaggle.registry import read_runs

DEFAULT_URI = "sqlite:///mlflow.db"
_TAG = "registry_run_id"
_MAX_PARAM = 6000  # MLflow's per-value limit


def _ms(stamp: str | None) -> int | None:
    if not stamp:
        return None
    return int(datetime.fromisoformat(stamp).timestamp() * 1000)


def _flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in params.items():
        full = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{full}."))
        else:
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            flat[full] = text[:_MAX_PARAM]
    return flat


def _tags(run: Mapping[str, Any]) -> dict[str, str]:
    git = run.get("git") or {}
    raw: dict[str, Any] = {
        _TAG: run["run_id"],
        "mlflow.runName": run.get("name"),
        "mlflow.source.git.commit": git.get("sha"),
        "git_dirty": git.get("dirty"),
        "git_diff_hash": git.get("diff_hash"),
        "command": " ".join(run.get("command") or []),
        "config_hash": run.get("config_hash"),
        "inputs": ",".join(run.get("inputs") or []),
        "artifacts": ",".join(run.get("artifacts") or []),
        "prereg": run.get("prereg"),
        "prereg_sha": run.get("prereg_sha"),
        "prereg_before_run": run.get("prereg_before_run"),
        "verdict": run.get("verdict"),
        "notes": run.get("notes"),
        "seed": run.get("seed"),
        "n_folds": run.get("n_folds"),
    }
    return {k: str(v)[:_MAX_PARAM] for k, v in raw.items() if v not in (None, "")}


def sync_mlflow(
    results_dir: Path,
    *,
    tracking_uri: str = DEFAULT_URI,
    experiment: str | None = None,
) -> tuple[int, int]:
    """Copy every registry run of ``results_dir`` into MLflow.

    Args:
        results_dir: A competition ``results/`` directory with ``runs.jsonl``.
        tracking_uri: MLflow backend, default a SQLite file in the cwd.
        experiment: Experiment name; defaults to the competition directory
            name (``results_dir``'s parent).

    Returns:
        ``(created, updated)`` run counts.
    """
    from mlflow.entities import Metric, Param, RunTag  # noqa: PLC0415 - optional extra
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    client = MlflowClient(tracking_uri=tracking_uri)
    exp_name = experiment or results_dir.resolve().parent.name
    exp = client.get_experiment_by_name(exp_name)
    exp_id = (
        exp.experiment_id if exp is not None else client.create_experiment(exp_name)
    )

    existing: dict[str, str] = {}
    token: str | None = None
    while True:
        page = client.search_runs([exp_id], max_results=1000, page_token=token)
        existing.update(
            {r.data.tags[_TAG]: r.info.run_id for r in page if _TAG in r.data.tags}
        )
        token = page.token
        if not token:
            break

    created = updated = 0
    for run in read_runs(results_dir):
        tags = _tags(run)
        if run["run_id"] in existing:
            mlflow_id = existing[run["run_id"]]
            client.log_batch(mlflow_id, tags=[RunTag(k, v) for k, v in tags.items()])
            updated += 1
            continue
        start, end = _ms(run.get("started_at")), _ms(run.get("finished_at"))
        mlflow_run = client.create_run(
            exp_id, start_time=start, tags=tags, run_name=run.get("name")
        )
        mlflow_id = mlflow_run.info.run_id
        stamp = end or start or 0
        metric = run.get("metric_name") or "score"
        metrics = []
        if run.get("oof_score") is not None:
            metrics.append(Metric(f"oof_{metric}", float(run["oof_score"]), stamp, 0))
        metrics += [
            Metric(f"fold_{metric}", float(s), stamp, i)
            for i, s in enumerate(run.get("fold_scores") or [], 1)
        ]
        params = [Param(k, v) for k, v in _flatten(run.get("params") or {}).items()]
        for i in range(0, max(len(params), len(metrics), 1), 100):  # MLflow batch limit
            client.log_batch(
                mlflow_id, metrics=metrics[i : i + 100], params=params[i : i + 100]
            )
        client.set_terminated(mlflow_id, status="FINISHED", end_time=end)
        created += 1
    return created, updated


def main(argv: Iterable[str] | None = None) -> int:
    """Sub-CLI behind ``python -m data_science_stuff.kaggle``."""
    parser = argparse.ArgumentParser(description="Sync results/runs.jsonl into MLflow")
    parser.add_argument("results_dirs", type=Path, nargs="+")
    parser.add_argument("--tracking-uri", default=DEFAULT_URI)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for results_dir in args.results_dirs:
        if not (results_dir / "runs.jsonl").exists():
            continue
        created, updated = sync_mlflow(results_dir, tracking_uri=args.tracking_uri)
        print(f"{results_dir}: {created} created, {updated} updated")
    return 0
