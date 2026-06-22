---
name: quality-gate
description: Run the repo's pre-commit quality gate (ruff format + check, mypy strict, pytest with 80% coverage) before committing or pushing Python changes. Use when asked to lint, format, type-check, run the checks, or verify code is ready to commit, e.g. "run the checks", "lint and type-check", "is this ready to push?".
---

# Pre-commit quality gate

Run this before committing or pushing any Python change. It mirrors the CI pipeline
(`.github/workflows/ci.yml`), so a clean local run means CI should pass.

## Steps (in order — stop and fix at the first failure)

1. **Format + autofix lint**
   ```bash
   make py-fmt        # ruff format . && ruff check --fix .
   ```
   Ruff config (`pyproject.toml` `[tool.ruff]`): line length 88, double quotes, Python 3.9
   target, broad ruleset (pyflakes, pycodestyle, bugbear, isort, bandit, pep8-naming, pyupgrade,
   …). `E501` and `PLR0913` are ignored. Some findings need manual fixes — `--fix` won't catch all.

2. **Strict type check**
   ```bash
   make py-static     # mypy . (strict mode)
   ```
   Every function needs type annotations. Use Python 3.9 **lowercase generics** (`list[int]`,
   `dict[str, object]`) — not `List`/`Dict` from `typing`. The `tests/` directory is exempt from
   strict mode. Prefer fixing types over scattering `# type: ignore`; when an ignore is
   unavoidable, give it a specific code (`# type: ignore[arg-type]`).

3. **Tests + coverage**
   ```bash
   pytest                      # full suite; coverage must stay ≥ 80% (--cov-fail-under=80)
   pytest -m "not slow"        # skip slow tests for a faster inner loop
   pytest -n auto              # parallel (pytest-xdist)
   pytest path/test_file.py::test_name   # single test
   ```
   Note: pytest runs with `--doctest-modules` and `--cov=data_science_stuff`. Mark long tests
   `@pytest.mark.slow` and integration tests `@pytest.mark.integration`.

## Notes

- All three commands assume the venv from `make dev-setup` is active (`source .venv/bin/activate`).
- Competition `src/` scripts are training/experiment code, not part of the coverage target
  (`--cov=data_science_stuff`); don't expect them to be exercised by the test suite.
- If `make py-fmt`/`make py-static` reformat or surface issues, re-run until clean before committing.
