# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A Python monorepo containing:
- **`data_mining/`** — KTH ID2222 coursework (Shingling/LSH, A-priori, Triest, Spectral Clustering, JaBeJa)
- **`implicit_neural_representations/`** — self-contained example (notebook + typed `src/` modules)
  of coordinate networks fitting an image (ReLU MLP vs Fourier features vs SIREN) and using the
  continuous representation for arbitrary-scale upscaling. Unlike the coursework/Kaggle dirs, its
  `src/` **is** held to the ruff/mypy gate. Website-bound content — keep the notebook narrative clean.
- **`playground-series-*/` / `playground_series_s5e1/`** — Kaggle Playground Series competition solutions
- **`data_science_stuff/`** — Shared installable package (versioned via git tags through
  setuptools_scm). `kaggle_utils.py` holds the original helpers (`save_cv_result`,
  `tune_decision_weights`, ...); the `kaggle/` subpackage holds the competition machinery
  extracted from s6e5–s6e10: `io` (paths/params/submissions), `device` (LightGBM CUDA probe,
  XGBoost device params, CatBoost 6 GB-GPU params), `cv` (`run_cv` fold loop, `run_cv_seeds`
  multi-seed bagging), `tuning` (`run_study` Optuna scaffold → `best_params*.json`),
  `blending` (Nelder-Mead blend weights, diversity report), `decision` (threshold weights —
  grid and Nelder-Mead fitters, bootstrap bagging, Bayes cost matrix, cascade recombination,
  XGBoost balanced-error eval metric), `encoding` (fold-safe target encoding, quantile bins,
  frequency features, rank-gauss, Fourier features), `stacking` (`stack_oof`, Caruana
  selection, per-base ridge calibration, public-notebook OOF loading), `compare` (seed-paired
  arm deltas, honest split-half blend reads), `screening` (the M-slope mechanism instrument,
  from s6e7), `shift` (adversarial validation, mask-mechanism diagnostics, uniform re-mask
  repair, from s6e7), `models` (MLP scaffold, RealMLP-TD, swap-noise DAE, imbalance losses),
  and the results layer: `predictions` (keyed parquet OOF/test files that carry `id` + `fold`),
  `registry` (append-only run registry `results/runs.jsonl`: git commit, command, params,
  config hash, lineage, verdicts), `prereg` (verifiable pre-registration commits) and
  `tracking` (optional MLflow view; `[tracking]` extra). CLI:
  `python -m data_science_stuff.kaggle {runs,prereg,mlflow-sync} ...`.

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
- **`ensemble-submit`** — rebuild the ensemble from saved OOF/test predictions (Nelder-Mead weights)
  and write a submission, enforcing the row-order invariant; covers stacking (LR-on-logits) when
  the scalar blend plateaus.
- **`select-finals`** — choose the final Kaggle submissions at competition end by CV rather than
  public LB rank, flagging public-split-overfit candidates.
- **`quality-gate`** — run `make py-fmt` → `make py-static` → `pytest` before committing.

A SessionStart hook (`.claude/hooks/session-start.sh`) installs dependencies automatically in
Claude Code on the web so the venv, linters, and tests are ready at session start.

## Shared code first

`data_science_stuff` is where reusable code lives. Two rules, both mandatory:

1. **Check the package before writing a helper.** Before adding any function to a competition
   script, search `data_science_stuff/` (`grep -rn "def " data_science_stuff/kaggle` or the
   module list above) for one that already does it, and import it. Re-written copies are the
   common failure: at the s6e10 audit there were 19 local `logit`/`softmax` copies
   (→ `stacking.clipped_logit`), 3 `inv_freq_weights` (→ `models.mlp.class_weights`),
   9 `ba`/`bacc` (→ `decision.balanced_accuracy_fast`) and 3 `weighted_bacc`
   (→ `shift.weighted_balanced_accuracy`). If the shared version is close but not quite
   right, extend it (a keyword argument with a backward-compatible default) rather than
   forking it locally.
2. **Promote code the second time it is needed.** When a helper is about to be copied into a
   second script of a different competition, or is imported by several scripts inside one
   competition (a `*_common.py`), move it into the package instead: generalise the
   competition-specific constants into arguments, add type hints (mypy strict) and tests
   in `tests/`, export it from the subpackage `__init__.py`, run the `quality-gate` skill,
   and update the module list in "What This Repo Is". Keep the competition's lesson in the
   docstring (what was measured, why the default is what it is), since it is the part that
   is expensive to rediscover.

Competition-specific feature engineering (`features.py`, domain keys, one-off probes) stays
in the competition directory. Finished competitions are not rewritten to use new shared
code; new scripts use it.

## Competition Workflow

Each Kaggle competition follows the same pipeline (mature references: `playground-series-s6e5/`
for regression/AUC, `playground-series-s6e6/` for imbalanced multiclass + stacking):

```
features.py  →  baseline.py / train_<model>.py  →  tune_<model>.py  →  ensemble.py  →  submission
        (5-fold stratified CV, results/oof_<m>.parquet + test_<m>.parquet)  (Nelder-Mead)   (kaggle CLI)
```

The pipeline is built on `data_science_stuff.kaggle`: `competition_dirs`/`load_params` →
`run_cv` (or `run_cv_seeds`; `run_study` in `tune_*.py`) → `save_predictions` + `save_cv_result` (registers the run) + `optimize_thresholds` →
`load_predictions` → `optimize_blend_weights`/`stack_oof` → `write_submission`. Competition
scripts import these; they never copy them. Older competitions (≤ s6e10) store positional
`oof_*.npy`; `load_predictions` still reads them (with a warning) and `migrate_npy` converts.

**What is persisted where:** `results/runs.jsonl` is the record of every run and is
**committed**; prediction parquet/npy files are gitignored (large); the competition `CLAUDE.md`
holds the reasoning (pre-registrations, readings) and cites run names. `make mlflow-ui` gives a
browsable view (`mlflow.db`, gitignored, rebuilt from the JSONL by `make mlflow-sync`).

Three invariants are critical and have each caused (or nearly caused) real failures:

1. **Row-order invariant** — `features.py::build_features()` sorts every dataframe by the
   competition's key columns. Save predictions with their ids (`save_predictions`) and load
   them by id (`load_predictions`), which raises on misalignment; legacy `oof_*.npy` arrays are
   positional in that sort order. Either way, `y` and ids **must** come from
   `build_features()`, or predictions silently misalign (a 0.5-AUC submission once resulted).
2. **GPU memory rule** — see below; omitting it OOM-crashes multi-fold/multi-trial runs.
3. **Fold-aware target encoding** — any target-derived feature is fit on the fold's training
   split only; fitting globally leaks val targets and inflates OOF scores that don't transfer
   to the leaderboard.

Per-competition `CLAUDE.md` files capture the dataset, EDA findings, current best, and an
experiments log; read them before working in a competition directory.

## Campaign lessons (earned, mostly on s6e7 — read before proposing a lever)

Late in a campaign the scores plateau and every candidate looks like ±0.0003. These are the
rules that separate the real ones from the rest. Each cost days to learn; none is obvious.

**Measuring a candidate**

1. **Screen by the decay exponent, not by per-seed significance.** Measure the paired delta at
   M = 1, 2, 3, 4 members and fit `|delta| ~ M**b`: `b > 0` is information (averaging *reveals*
   it), `b ≈ -0.5` is selection noise, `b ≈ -1` is variance reduction and is dead at any
   deployed breadth. "Positive at every seed with a tiny standard error" is exactly what a
   worthless lever looks like — a variance reducer is significantly positive at every single
   member by construction. Use `kaggle.screening.screen_mechanism`; derivation and the measured
   controls are in `playground-series-s6e7/notes/mslope_writeup.md`.
2. **Calibrate the instrument before quoting it.** Magnitude-match the known-real and known-fake
   controls (`mean_shift_control` / `variance_reduction_control`) on your own data; the
   selection-noise band is instrument-dependent and must never be imported from another
   competition.
3. **Fit the decision layer honestly, per arm.** When challenger and stock are two *different*
   feature matrices their posteriors are calibrated differently, and freezing one arm's weights
   across both flips signs. Fit on one half, score on the other, both directions.
4. **Do not fit an exponent (or any slope) on three points.** s6e7 published a `+0.475` from a
   3-point fit and had to correct it to `-0.158` at six.
5. **A step-function objective needs a grid, not a simplex.** `balanced_accuracy(argmax(p·w))`
   is piecewise constant; Nelder-Mead cannot converge on it and burns its full budget every
   call. `decision.tune_decision_weights_grid` is ~30× faster *and* found a better optimum.
6. **OOF cannot rank below its noise floor.** Under covariate shift (`shift.adversarial_validation`
   AUC ≥ 0.6) absolute CV↔LB calibration can stay excellent while the *ordering* of
   sub-0.001 candidates is noise. Trust OOF for coarse decisions only.

**Where a lever has to pay**

7. **A strong combiner absorbs leg-level gains — added legs *and* improved legs.** An LR stack
   over multi-seed breadths is itself an averaging operator, so it already extracts anything
   that works by averaging or by a small mean shift: s6e7 measured a leg improved by +0.00045
   solo delivering +0.00001 to the stack. Only a leg that differs *in kind* moves a stack.
   Price every lever at the deliverable, never at the leg.

**Data**

8. **Check the missingness *mechanism*, not the missing rate.** s6e7's train masks were
   deterministic trigger rules and its test masks were uniform *at identical marginal rates* —
   invisible to every rate-level check, and worth real score. `shift.mask_mechanism_report`
   compares per-column MAR AUCs across splits; `shift.unseen_mask_patterns` counts test rows
   whose pattern is impossible in train.
9. **Surfaces do not mix.** If you repair a train/test discrepancy by altering the train matrix,
   every OOF array produced that way lives on a different surface, reads systematically lower,
   and may only be compared with — or stacked against — arrays from the same surface.

**Process**

10. **Pre-register the bar before the number lands, and do not move it afterwards.** Write the
    arm, the instrument, the prior, and the pass/fail threshold into the competition CLAUDE.md
    *before* running, and commit it with `python -m data_science_stuff.kaggle prereg PR-<n>
    <competition>/CLAUDE.md` — a registration that shares a commit with its result cannot prove
    which came first. Register the run with `prereg="PR-<n>"` and the registry records whether
    the commit predates it. State what the honest expected outcome is; a clean negative against
    an unmoved bar is a successful experiment.
11. **Verify inherited facts at their source.** An unsourced deadline propagated through 22 days
    of s6e7 planning and was wrong by two days — which silently deleted the "select finals ≥48 h
    early" margin the whole endgame protocol assumed.
12. **A cache key must encode every dimension that varies.** A key of `{tag}_s{seed}_f{fold}`
    that omits, say, which driver is being imputed will silently serve a stale block for the
    second one.
13. **Flat closes an experiment, not an avenue.** Do not report a plateau as a ceiling; say what
    was measured and what class of lever remains untested.

**The generator (lost s6e10 to it; Deotte won S6E9 and S6E10 with it)**

14. **Model the generator, in its own model family.** Playground data is sampled by GReaT
    (Borisov et al. 2023): GPT-2 fine-tuned on the original rows as `"col is value, ..."` text
    in random column order, sampled at about temperature 0.7 / top-k 50, **label generated
    first**. Check it on day 1: with the GPT-2 tokenizer, 3-digit numbers are one token and
    4-digit numbers are two, and a label-first prompt reproduces the train label rate. The
    winning leg is a label-first GPT-2 "Storyteller":
    - Fine-tune a GReaT replica on the original data (about 60 epochs).
    - Continue per fold on train, label first, columns shuffled each step (8 epochs, lr 5e-5,
      bf16).
    - Score each row as `log P(x | y=1) − log P(x | y=0)` with logsumexp over about 30 column
      orders, then add the prior log-odds.

    In s6e10 that one leg scored 0.9624 OOF, above our whole 364-member stack (0.96224), and
    added +5e-4 CV to the public pool. We had found that a generative leg was the missing kind
    (s6e10 PR-51/57) but built it from trees (0.954 solo). Write values the way the original
    data writes them (`36`, `0.0`, `None`, `True`/`False`). Budget about 2 A100-hours per fold,
    so plan Kaggle or cloud GPUs early, because 6 GB local is too slow. A label-last GPT-2
    ("Detective", about 0.961) is a cheaper second leg that makes different errors.
    Training with the label at a random position ("label anywhere") gives one model that reads both ways and was
    Deotte's best s6e10 single model (0.96245). Replicate the generator in its own pinned env (be-great 0.0.3,
    transformers 4.26.1), at several epoch counts, and note that it was trained on a noised copy of the original
    (~7% label flips in his reconstruction).

**What the winner's public code showed (`notes/deotte-playground-solutions.md`, read it at competition start)**

On s6e5, s6e8 and s6e10, same folds, Deotte's best *single* model beat our best ensemble: 0.9544 vs 0.9508,
0.9707 vs 0.9661, 0.96245 vs 0.96227. The gap was in how the data was made, not in measurement or stacking.

15. **There are two generators. Reverse-engineer both on day 1.** Many original datasets were produced by a script
    with a label rule or arithmetic identities (s6e9: `Phi((s - 5.6092) / 0.8345)`; s6e8: `daily > 8 or social > 4`;
    s6e3: `TotalCharges - tenure * MonthlyCharges`). Fit the rule on the *original* rows, not on the synthetic ones,
    whose noise hides it. Use the rule as a base margin, as an offset, and as the target of residual encodings.
16. **Use the original data in every model.** Our s6e5 and s6e8 never downloaded it. Four uses: append the original
    rows to each fold's training side with a tuned weight; per-value and per-pair smoothed LLRs fitted on the original;
    snap features (nearest original value and the difference); and per-class ECDF and nearest-original distance.
17. **Synthetic values are a finite vocabulary.** Give each numeric an exact-value categorical token as well as the
    value. Add digit, modulus and decimal artifact columns, and for GReaT data the GPT-2 BPE pieces of the value
    string (about +0.0002 per XGBoost in s6e9). Label-free density models over train + test (masked-LM completion
    of missing cells as a full conditional distribution, reconstruction of each field from the others, surprisal)
    are legal and were s6e8's best channels.
18. **A weak MLP is a bug, not a verdict.** pytabkit `RealMLP_TD` with exact-value tokens and nested TEs was the best
    single model in s6e5 and s6e8, where our MLPs earned weight 0. His recipe: 32 internal members, epoch picked on
    a 15% inner split then refit on all outer-train rows, label smoothing 0.005, 3-5 seeds per fold.
19. **Never early-stop on the validation fold.** Stop on an inner 10% holdout, then refit on all outer-train rows for
    about 1.1x the rounds. Give GBDTs a `base_margin` / `init_score` from an in-fold LR, the recovered rule or a
    TE-WOE sum.
20. **Run TFMs zero-shot on raw features at full context** (s6e10: TabPFN-3.5 0.9611 and Kumo-large 0.9600 beat
    tuned XGBoost). They need 20-84 GB of GPU memory, and the generator legs need A100-class GPUs. Rent that compute
    in week 1. Do not discover in the last week that the 6 GB card cannot run them.
21. **Spend effort where the gap can be, not where it can be measured.** We spent s6e10 PR-43 to PR-60 on
    ±0.00001 levers while a 0.0006 lever was untried. Breadth of model *classes* (best one of each) matters, the
    count of models does not. Keep every experiment's OOF/test pair and let an L2 LR on logits weigh them.
    Pre-registration still governs small effects. It cannot find the large effects nobody listed.

**The public notebooks (s6e11: +0.001 LB in 20 minutes once we looked)**

15. **Read the public notebooks on day 1 and every ~3 days, and pool their OOF banks.**
    Working LB-blind means not chasing LB deltas. It does not mean ignoring what the public
    has found. On s6e11 we spent two days on our own pipeline at LB 0.95110 while the top was
    at 0.95229. The notebooks had a feature family we never tried: target encoding of the
    float columns' *digit keys* (integer part, fraction, leading digits, GPT-2 tokens). That
    gave a single shallow tree 0.9515 OOF against our 0.948. They also published OOF banks on
    the community folds (`StratifiedKFold(5, shuffle=True, random_state=42)` on id-sorted
    rows, which is also our `fold_assignment`). Stacking our different-kind leg (the
    label-first GPT-2) on a public bank took LB from 0.95110 to 0.95212. The routine:
    - Compare our best public score with the top: `kaggle competitions leaderboard <c> -s`.
    - List the notebooks: `kaggle kernels list --competition <c> --sort-by voteCount`, then
      `--sort-by scoreDescending`. Scan the titles for feature families we don't know.
    - Read the top ones: `kaggle kernels pull <ref>`.
    - Download their OOF banks: `kaggle kernels output <ref>`, or the dataset they attach.
    - Stack the banks with our legs (s6e11 `src/ext_stack.py`: LR on logit-ranks, with nested
      CV for the stacker).
    - Price our own levers *inside the pooled stack*: legs that helped our stack added nothing
      next to the bank (s6e11: our EBM/trees 0, ep20 LM 0). Only a leg that differs in kind
      survives (lesson 7).

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
