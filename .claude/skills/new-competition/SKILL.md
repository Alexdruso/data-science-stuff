---
name: new-competition
description: Scaffold a new Kaggle Playground Series competition directory in this monorepo, following the repo's standard structure (data/, src/, notebooks/, submissions/, results/) and seeding features.py, cv_results.py, baseline.py, and a competition CLAUDE.md. Use when starting work on a new Kaggle competition, e.g. "set up playground-series-s6e7", "scaffold a new competition", "start the s7e1 competition".
---

# Scaffold a new Kaggle Playground competition

Use this when the user wants to begin a new Kaggle Playground Series competition. The goal is
to reproduce the structure of the mature reference competition `playground-series-s6e5/` so the
features → train/tune → ensemble → submit pipeline works immediately.

## Inputs to confirm first

- **Competition id / slug** (e.g. `playground-series-s6e7`) — used for the directory name and
  the `kaggle competitions download -c <id>` call.
- **Task type** (binary/multiclass classification, regression) and **metric** (AUC, RMSE, …).
- **Target column name** and the **row-order invariant keys** (the columns the dataset must be
  sorted by so OOF/test arrays stay aligned — for s6e5 these were
  `["Driver", "Race", "Year", "LapNumber"]`). If unknown, inspect the CSV headers after download.

## Steps

1. **Create the directory tree** at the repo root:
   ```
   <id>/
   ├── data/          # gitignored CSVs (ensure data/ is covered by .gitignore)
   ├── src/
   ├── notebooks/
   ├── submissions/
   ├── results/
   ├── README.md
   └── CLAUDE.md
   ```

2. **Download the data** (the `kaggle` CLI is a project dependency and `Bash(kaggle *)` is
   already allow-listed):
   ```bash
   kaggle competitions download -c <id> -p <id>/data/
   cd <id>/data && unzip -q <id>.zip && rm <id>.zip && cd -
   ```
   Confirm `data/` is gitignored so the CSVs are never committed.

3. **Seed `src/`** by adapting the reference files (do not invent new conventions). All shared
   machinery is imported from `data_science_stuff.kaggle` — **import, never copy** season
   scripts (the subpackage covers io / device / cv / blending / decision / encoding / stacking /
   models; the season files are worked examples of using it):
   - `src/cv_results.py` — a thin re-export of the **shared** helper, copied verbatim from
     `playground-series-s6e5/src/cv_results.py`:
     `from data_science_stuff.kaggle_utils import save_cv_result`.
   - `src/features.py` — adapt `playground-series-s6e5/src/features.py`. Keep the
     **`build_features(df)` contract**: it must `df.sort([...invariant keys...])` FIRST, then add
     derived columns, and return a Polars DataFrame. Define `TARGET` and `DRIVER_COLS` (or the
     equivalent set of columns to exclude from the model) at module top.
   - `src/baseline.py` — adapt `playground-series-s6e6/src/baseline.py` (the fully
     package-driven example): LightGBM with
     `from data_science_stuff.kaggle.device import get_lgbm_device`, paths via
     `competition_dirs(__file__)`, tuned params via `load_params(RESULTS_DIR, DEFAULTS, ...)`,
     the fold loop via `run_cv(X, y, X_test, fit_fold, ...)` (stratified seed-42 by default;
     `stratified=False` for regression), OOF/test saved as `results/oof_lgbm.npy` +
     `results/test_lgbm.npy`, logging via `save_cv_result`, submission via
     `write_submission(SUBMISSIONS_DIR, name, test_ids, TARGET, preds)`.

4. **Write `CLAUDE.md`** from `.claude/templates/competition-CLAUDE.md`, filling the
   placeholders (task, metric, target, invariant keys, deadline). The row-order invariant warning
   at the top must name THIS competition's sort keys.

5. **Write a short `README.md`** (one-line description + link to the Kaggle competition page).

## Conventions to preserve

- Polars for I/O and feature engineering; convert to pandas only at model-fit time
  (`.to_pandas()`, cast string categoricals to `category`).
- Paths: `DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)` from
  `data_science_stuff.kaggle.io` — never hand-build the `Path(__file__).parent.parent` trio.
- Cross-module imports use `sys.path.insert(0, str(Path(__file__).parent))` for the
  competition's own modules (`from features import build_features`); shared utilities come
  from `data_science_stuff.kaggle` as normal package imports.
- Type hints everywhere (mypy strict); Python 3.9 lowercase generics (`list[int]`, `dict[str, object]`).

## Verify

Run `cd <id> && python src/baseline.py` (requires the venv from `make dev-setup`). It should
print per-fold scores, an OOF score, and write `results/oof_lgbm.npy`, `results/test_lgbm.npy`,
and a submission CSV. Then run `make py-fmt && make py-static` (see the `quality-gate` skill).
