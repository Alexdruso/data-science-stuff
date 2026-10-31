"""Tests for the run registry, pre-registration commits and the MLflow mirror."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_science_stuff.kaggle import prereg, registry
from data_science_stuff.kaggle._git import git
from data_science_stuff.kaggle.prereg import commit_prereg, find_prereg
from data_science_stuff.kaggle.registry import (
    annotate_run,
    config_hash,
    git_state,
    import_cv_scores,
    load_runs,
    log_run,
    read_runs,
)
from data_science_stuff.kaggle_utils import save_cv_result


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo shaped like a competition directory."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var, val in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(var, val)
    root = tmp_path / "repo"
    comp = root / "playground-series-x"
    (comp / "results").mkdir(parents=True)
    (comp / "src").mkdir()
    (comp / "CLAUDE.md").write_text("# x\n")
    (comp / "src" / "train.py").write_text("print(1)\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)  # noqa: S603, S607
    subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "commit", "-q", "-m", "init"],  # noqa: S607
        check=True,
    )
    return comp


# --------------------------------------------------------------------- registry


def test_config_hash_is_order_insensitive() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_log_run_records_everything(repo: Path) -> None:
    results = repo / "results"
    rec = log_run(
        results,
        "lgbm_v1",
        oof_score=0.95,
        fold_scores=[0.94, np.float64(0.96)],
        metric_name="auc",
        params={"lr": np.float32(0.05), "leaves": 31, "path": Path("x")},
        inputs=["realmlp_v1"],
        artifacts=[results / "oof_lgbm_v1.parquet"],
        seed=42,
        command=["python", "src/train.py", "--route"],
    )
    assert rec.n_folds == 2
    assert rec.git.sha is not None
    assert rec.git.dirty is False
    assert rec.params["lr"] == pytest.approx(0.05)
    assert rec.params["path"] == "x"

    (run,) = read_runs(results)
    assert run["run_id"] == rec.run_id
    assert run["fold_scores"] == [0.94, 0.96]
    assert run["inputs"] == ["realmlp_v1"]
    assert run["command"][-1] == "--route"

    df = load_runs(results)
    assert df.loc[0, "git_dirty"] is False or df.loc[0, "git_dirty"] == False  # noqa: E712
    assert df.loc[0, "oof_score"] == 0.95


def test_config_hash_tracks_params_command_and_inputs(repo: Path) -> None:
    results = repo / "results"
    a = log_run(results, "m", params={"x": 1}, command=["c"])
    b = log_run(results, "m", params={"x": 1}, command=["c"])
    c = log_run(results, "m", params={"x": 2}, command=["c"])
    d = log_run(results, "m", params={"x": 1}, command=["c"], inputs=["other"])
    assert a.config_hash == b.config_hash
    assert a.run_id != b.run_id
    assert len({a.config_hash, c.config_hash, d.config_hash}) == 3


def test_git_state_scoped_to_paths(repo: Path) -> None:
    root = repo.parent
    (root / "elsewhere.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(root), "add", "elsewhere.txt"], check=True)  # noqa: S603, S607
    assert git_state([repo]).dirty is False  # staged change outside the competition

    (repo / "src" / "train.py").write_text("print(2)\n")
    state = git_state([repo])
    assert state.dirty is True
    assert state.dirty_paths == ["playground-series-x/src/train.py"]
    first_hash = state.diff_hash
    (repo / "src" / "train.py").write_text("print(3)\n")
    assert git_state([repo]).diff_hash != first_hash


def test_git_state_outside_repo(tmp_path: Path) -> None:
    state = git_state([tmp_path])
    assert state.sha is None
    assert state.dirty is None


def test_nan_score_stored_as_null(repo: Path) -> None:
    log_run(repo / "results", "m", oof_score=float("nan"))
    assert read_runs(repo / "results")[0]["oof_score"] is None


def test_annotate_and_corrupt_lines(repo: Path) -> None:
    results = repo / "results"
    rec = log_run(results, "m", oof_score=0.9)
    annotate_run(results, rec.run_id, verdict="FAIL", notes="sub-bar")
    annotate_run(results, rec.run_id, verdict="PASS")
    with (results / registry.RUNS_FILE).open("a") as f:
        f.write('{"event": "run", "run_id": "trunc')  # machine died mid-write
    with pytest.warns(UserWarning, match="unreadable"):
        (run,) = read_runs(results)
    assert run["verdict"] == "PASS"
    assert run["notes"] == "sub-bar"
    with pytest.raises(KeyError):
        annotate_run(results, "nope", verdict="x")


def test_read_runs_missing_file(tmp_path: Path) -> None:
    assert read_runs(tmp_path) == []
    assert load_runs(tmp_path).empty


def test_save_cv_result_registers(repo: Path) -> None:
    results = repo / "results"
    run_id = save_cv_result(
        results,
        "xgb_v1",
        [0.9, 0.91],
        0.905,
        metric_name="auc",
        params={"d": 6},
        seed=1,
    )
    assert run_id is not None
    (run,) = read_runs(results)
    assert run["run_id"] == run_id
    assert run["metric_name"] == "auc"
    assert run["params"] == {"d": 6}
    assert save_cv_result(results, "xgb_v2", [0.9], 0.9, register=False) is None
    assert len(read_runs(results)) == 1
    assert len(pd.read_csv(results / "cv_scores.csv")) == 2


def test_import_cv_scores_is_idempotent(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        {
            "model": ["a", "b"],
            "timestamp": ["2026-10-05 14:20:41", "2026-10-05 15:00:00"],
            "oof_auc": [0.95, 0.96],
            "fold_1_auc": [0.94, 0.95],
            "fold_2_auc": [0.96, np.nan],
        }
    ).to_csv(results / "cv_scores.csv", index=False)
    assert import_cv_scores(results) == 2
    assert import_cv_scores(results) == 0
    runs = read_runs(results)
    assert [r["name"] for r in runs] == ["a", "b"]
    assert runs[1]["fold_scores"] == [0.95]
    assert runs[0]["metric_name"] == "auc"
    assert runs[0]["started_at"].startswith("2026-10-05T14:20:41")


def test_import_cv_scores_bad_and_missing(tmp_path: Path) -> None:
    assert import_cv_scores(tmp_path) == 0
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "cv_scores.csv", index=False)
    with pytest.raises(ValueError, match="model"):
        import_cv_scores(tmp_path)


def test_registry_cli(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = repo / "results"
    assert registry.main(["list", str(results)]) == 0
    assert "no runs" in capsys.readouterr().out
    log_run(results, "m", oof_score=0.9)
    assert registry.main(["list", str(results), "-n", "0"]) == 0
    assert "m-" in capsys.readouterr().out
    assert registry.main(["import-cv-scores", str(results)]) == 0
    assert "imported 0" in capsys.readouterr().out


# ---------------------------------------------------------------------- prereg


def _register(repo: Path, pr_id: str) -> prereg.PreregCommit:
    md = repo / "CLAUDE.md"
    md.write_text(md.read_text() + f"\n### {pr_id} — bar +0.0005\n")
    return commit_prereg(pr_id, md, message="bar +0.0005")


def test_prereg_before_run_is_verified(repo: Path) -> None:
    found = _register(repo, "PR-5")
    assert find_prereg("PR-5", cwd=repo) == found
    assert find_prereg("PR-55", cwd=repo) is None  # exact subject match only
    assert git("log", "-1", "--format=%s", cwd=repo) == "prereg: PR-5"

    later = found.committed_at + timedelta(seconds=5)
    rec = log_run(repo / "results", "m", prereg="PR-5", started_at=later)
    assert rec.prereg_sha == found.sha
    assert rec.prereg_before_run is True

    earlier = found.committed_at - timedelta(hours=1)
    rec2 = log_run(repo / "results", "m", prereg="PR-5", started_at=earlier)
    assert rec2.prereg_before_run is False


def test_prereg_earliest_commit_counts(repo: Path) -> None:
    first = _register(repo, "PR-7")
    _register(repo, "PR-7")  # an amendment after the fact
    assert find_prereg("PR-7", cwd=repo) == first


def test_unregistered_prereg_warns(repo: Path) -> None:
    with pytest.warns(UserWarning, match="NOT pre-registered"):
        rec = log_run(repo / "results", "m", prereg="PR-99")
    assert rec.prereg_before_run is False
    assert rec.prereg_sha is None


def test_commit_prereg_commits_only_claude_md(repo: Path) -> None:
    (repo / "src" / "train.py").write_text("print('wip')\n")
    root = repo.parent
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)  # noqa: S603, S607
    found = _register(repo, "PR-8")
    changed = git("show", "--name-only", "--format=", found.sha, cwd=repo)
    assert changed == "playground-series-x/CLAUDE.md"
    staged = git("diff", "--cached", "--name-only", cwd=repo)
    assert staged == "playground-series-x/src/train.py"  # still staged, not committed


def test_commit_prereg_guards(repo: Path, tmp_path: Path) -> None:
    md = repo / "CLAUDE.md"
    with pytest.raises(ValueError, match="no uncommitted changes"):
        commit_prereg("PR-1", md)
    md.write_text(md.read_text() + "\nsomething else\n")
    with pytest.raises(ValueError, match="do not mention"):
        commit_prereg("PR-1", md)
    with pytest.raises(ValueError, match="invalid"):
        commit_prereg("PR 1; rm", md)
    outside = tmp_path / "loose.md"
    outside.write_text("PR-1\n")
    with pytest.raises(RuntimeError, match="git diff failed"):
        commit_prereg("PR-1", outside)


def test_prereg_cli(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md = repo / "CLAUDE.md"
    md.write_text(md.read_text() + "\n### PR-60\n")
    assert prereg.main(["PR-60", str(md), "-m", "bar"]) == 0
    assert "registered PR-60" in capsys.readouterr().out


def test_find_prereg_outside_repo(tmp_path: Path) -> None:
    assert find_prereg("PR-1", cwd=tmp_path) is None


# ---------------------------------------------------------------------- mlflow


def test_sync_mlflow_is_idempotent(repo: Path, tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    from data_science_stuff.kaggle import tracking  # noqa: PLC0415 - optional extra

    results = repo / "results"
    rec = log_run(
        results,
        "lgbm",
        oof_score=0.95,
        fold_scores=[0.94, 0.96],
        metric_name="auc",
        params={"lr": 0.05, "nested": {"a": 1}},
        inputs=["x", "y"],
    )
    log_run(results, "bare")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert tracking.sync_mlflow(results, tracking_uri=uri) == (2, 0)
    annotate_run(results, rec.run_id, verdict="FAIL")
    assert tracking.sync_mlflow(results, tracking_uri=uri) == (0, 2)

    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("playground-series-x")
    assert exp is not None
    runs = {
        r.data.tags["registry_run_id"]: r
        for r in client.search_runs([exp.experiment_id])
    }
    run = runs[rec.run_id]
    assert run.data.metrics["oof_auc"] == 0.95
    assert run.data.params["nested.a"] == "1"
    assert run.data.tags["verdict"] == "FAIL"
    assert run.data.tags["inputs"] == "x,y"
    history = client.get_metric_history(run.info.run_id, "fold_auc")
    assert [m.step for m in history] == [1, 2]
    assert tracking.main([str(results), str(tmp_path), "--tracking-uri", uri]) == 0


def test_tracking_ms_handles_offsets() -> None:
    pytest.importorskip("mlflow")
    from data_science_stuff.kaggle.tracking import _ms  # noqa: PLC0415

    stamp = datetime(2026, 10, 31, 12, tzinfo=timezone.utc).isoformat()
    assert _ms(stamp) == 1793448000000
    assert _ms("") is None
    assert json.dumps(_ms(None)) == "null"


def test_package_cli_dispatch(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from data_science_stuff.kaggle.__main__ import main  # noqa: PLC0415

    assert main([]) == 0
    assert "mlflow-sync" in capsys.readouterr().out
    assert main(["bogus"]) == 2
    assert main(["runs", "list", str(repo / "results")]) == 0
    md = repo / "CLAUDE.md"
    md.write_text(md.read_text() + "\n### PR-61\n")
    assert main(["prereg", "PR-61", str(md)]) == 0
    if importlib.util.find_spec("mlflow") is not None:
        assert main(["mlflow-sync", str(repo / "results")]) == 0
