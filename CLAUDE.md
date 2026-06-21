# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A Python monorepo containing:
- **`data_mining/`** — KTH ID2222 coursework (Shingling/LSH, A-priori, Triest, Spectral Clustering, JaBeJa)
- **`playground-series-*/` / `playground_series_s5e1/`** — Kaggle Playground Series competition solutions
- **`data_science_stuff/`** — Shared installable package (versioned via git tags through setuptools_scm)

## Setup

```bash
make dev-setup          # Creates .venv, installs uv, installs all dependencies
source .venv/bin/activate
```

To install manually: `uv pip install -e ".[development,docs]"`

## Common Commands

```bash
make py-fmt             # ruff format + ruff check --fix
make py-static          # mypy strict type checking
make clean              # remove .venv and all caches

pytest                                          # run all tests
pytest tests/path/test_file.py::test_name      # single test
pytest -m "not slow"                            # skip slow tests
pytest -n auto                                  # parallel execution
pytest --cov=data_science_stuff --cov-report=term-missing
```

## Code Quality

**Ruff** (line length 88, double quotes, Python 3.9 target) runs a broad ruleset including pyflakes, pycodestyle, bugbear, isort, bandit, pep8-naming, pyupgrade, and more. E501 and PLR0913 are ignored.

**MyPy** runs in strict mode — all functions require type annotations. Tests directory is exempt from strict mode.

**Coverage minimum**: 80% (enforced in CI).

## CI Pipeline (`.github/workflows/ci.yml`)

Matrix: Python 3.9, 3.10, 3.11 on Ubuntu. Two jobs:
1. **lint-and-test**: ruff → mypy → bandit → safety → pytest → codecov
2. **security**: Snyk vulnerability scan

## Claude Code Skills

Project skills live in `.claude/skills/`. Invoke the matching one instead of re-deriving these
workflows by hand:

- **`new-competition`** — scaffold a new Kaggle Playground competition (standard directory tree,
  download data via the `kaggle` CLI, seed `features.py`/`cv_results.py`/`baseline.py` and a
  competition `CLAUDE.md` from `.claude/templates/competition-CLAUDE.md`).
- **`add-model`** — add a `train_<model>.py` (+ optional `tune_<model>.py`) that follows the
  CV/OOF conventions and the PyTorch GPU memory rule.
- **`ensemble-submit`** — rebuild the ensemble from `oof_*.npy`/`test_*.npy` (Nelder-Mead weights)
  and write a submission, enforcing the row-order invariant.
- **`quality-gate`** — run `make py-fmt` → `make py-static` → `pytest` before committing.

A SessionStart hook (`.claude/hooks/session-start.sh`) installs dependencies automatically in
Claude Code on the web so the venv, linters, and tests are ready at session start.

## Competition Workflow

Each Kaggle competition follows the same pipeline (mature reference: `playground-series-s6e5/`):

```
features.py  →  baseline.py / train_<model>.py  →  tune_<model>.py  →  ensemble.py  →  submission
            (5-fold stratified CV, results/oof_<m>.npy + test_<m>.npy)  (Nelder-Mead)   (kaggle CLI)
```

Two invariants are critical and have each caused real failures:

1. **Row-order invariant** — `features.py::build_features()` sorts every dataframe by the
   competition's key columns, and all `oof_*.npy`/`test_*.npy` arrays are stored in that order.
   Any code that loads `y` or `test_ids` to combine with those arrays **must** go through
   `build_features()`, or predictions silently misalign (a 0.5-AUC submission once resulted).
2. **GPU memory rule** — see below; omitting it OOM-crashes multi-fold/multi-trial runs.

Per-competition `CLAUDE.md` files capture the dataset, EDA findings, current best, and an
experiments log; read them before working in a competition directory.

## Architecture Conventions

### Data Science Preferences (from `.cursor/rules/`)
- **Polars over Pandas** for performance-sensitive work (Pandas still present for compatibility)
- **PyCaret** for rapid prototyping and baseline models before custom implementations
- Proper cross-validation; document experiments and hyperparameters

### Kaggle Competition Structure
Each competition directory follows: `data/` (gitignored CSVs), `src/` (modules), `notebooks/` (EDA/experiments), `submissions/`.

### Type Hints
Required everywhere (MyPy strict). Use `dataclasses` for configuration objects. Python 3.9 target — use `list[int]`, `dict[str, int]` (lowercase generics) rather than `List`, `Dict` from `typing`.

### PyTorch GPU memory
Every function that allocates GPU tensors or models must `del` them before returning, and the outer fold/trial loop must call `torch.cuda.empty_cache()` after each iteration. This applies to both `train_*.py` and `tune_*.py` scripts. Omitting this in a multi-fold/multi-trial run causes CUDA memory fragmentation and OOM crashes.

### Testing
Tests go in `tests/`. Mark long-running tests with `@pytest.mark.slow`, integration tests with `@pytest.mark.integration`.

## Versioning

`data_science_stuff/_version.py` is auto-generated from git tags via `setuptools_scm` (semver tags like `v1.0.0`). Do not edit it manually; it is gitignored.
