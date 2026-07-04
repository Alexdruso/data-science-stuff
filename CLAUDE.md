# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A Python monorepo containing:
- **`data_mining/`** — KTH ID2222 coursework (Shingling/LSH, A-priori, Triest, Spectral Clustering, JaBeJa)
- **`playground-series-*/` / `playground_series_s5e1/`** — Kaggle Playground Series competition solutions
- **`data_science_stuff/`** — Shared installable package (versioned via git tags through
  setuptools_scm). `kaggle_utils.py` holds the original helpers (`save_cv_result`,
  `tune_decision_weights`, ...); the `kaggle/` subpackage holds the competition machinery
  extracted from s6e5/s6e6: `io` (paths/params/submissions), `device` (LightGBM CUDA probe),
  `cv` (`run_cv` fold loop), `blending` (Nelder-Mead blend weights, diversity report),
  `decision` (threshold weights, Bayes cost matrix, cascade recombination), `encoding`
  (fold-safe target encoding, quantile bins, frequency features), `stacking` (`stack_oof`,
  Caruana selection), and `models` (MLP scaffold, RealMLP-TD, imbalance losses).

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

Matrix: Python 3.9, 3.10, 3.11 on Ubuntu. One job:
1. **lint-and-test**: ruff → mypy → bandit → pytest → codecov

**Gate scope**: ruff, mypy, and bandit are intentionally scoped to the maintained package
(`data_science_stuff/`) and its `tests/`. The Kaggle competition (`playground-*`,
`playground_*`) and `data_mining/` coursework directories are experiment code and are excluded
from the gate (ruff/mypy via `pyproject.toml`, bandit via its explicit target in `ci.yml`). The
exclusion globs cover competitions scaffolded in the future, so new competition dirs don't
re-break CI.

## Claude Code Skills

Project skills live in `.claude/skills/`. Invoke the matching one instead of re-deriving these
workflows by hand:

- **`new-competition`** — scaffold a new Kaggle Playground competition (standard directory tree,
  download data via the `kaggle` CLI, seed `features.py`/`cv_results.py`/`baseline.py` and a
  competition `CLAUDE.md` from `.claude/templates/competition-CLAUDE.md`).
- **`add-model`** — add a `train_<model>.py` (+ optional `tune_<model>.py`) that follows the
  CV/OOF conventions and the PyTorch GPU memory rule.
- **`ensemble-submit`** — rebuild the ensemble from `oof_*.npy`/`test_*.npy` (Nelder-Mead weights)
  and write a submission, enforcing the row-order invariant; covers stacking (LR-on-logits) when
  the scalar blend plateaus.
- **`select-finals`** — choose the final Kaggle submissions at competition end by CV rather than
  public LB rank, flagging public-split-overfit candidates.
- **`quality-gate`** — run `make py-fmt` → `make py-static` → `pytest` before committing.

A SessionStart hook (`.claude/hooks/session-start.sh`) installs dependencies automatically in
Claude Code on the web so the venv, linters, and tests are ready at session start.

## Competition Workflow

Each Kaggle competition follows the same pipeline (mature references: `playground-series-s6e5/`
for regression/AUC, `playground-series-s6e6/` for imbalanced multiclass + stacking):

```
features.py  →  baseline.py / train_<model>.py  →  tune_<model>.py  →  ensemble.py  →  submission
            (5-fold stratified CV, results/oof_<m>.npy + test_<m>.npy)  (Nelder-Mead)   (kaggle CLI)
```

The pipeline is built on `data_science_stuff.kaggle`: `competition_dirs`/`load_params` →
`run_cv` → `save_cv_result` + `optimize_thresholds` → `optimize_blend_weights`/`stack_oof` →
`write_submission`. Competition scripts import these; they never copy them.

Three invariants are critical and have each caused (or nearly caused) real failures:

1. **Row-order invariant** — `features.py::build_features()` sorts every dataframe by the
   competition's key columns, and all `oof_*.npy`/`test_*.npy` arrays are stored in that order.
   Any code that loads `y` or `test_ids` to combine with those arrays **must** go through
   `build_features()`, or predictions silently misalign (a 0.5-AUC submission once resulted).
2. **GPU memory rule** — see below; omitting it OOM-crashes multi-fold/multi-trial runs.
3. **Fold-aware target encoding** — any target-derived feature is fit on the fold's training
   split only; fitting globally leaks val targets and inflates OOF scores that don't transfer
   to the leaderboard.

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
