---
name: add-model
description: Add a new model to a Kaggle competition in this monorepo — a train_<model>.py (and optional tune_<model>.py) that follows the repo's CV/OOF conventions and GPU memory rule. Use when asked to add or train a new model, e.g. "add a CatBoost model to s6e5", "write an XGBoost trainer", "add a tuning script for the MLP".
---

# Add a model to a competition

Use this to add a new base learner to an existing competition (e.g. `playground-series-s6e5/`).
Every model must plug into the ensemble, so the OOF/test array contract and CV setup are
non-negotiable. Mirror the existing reference scripts rather than inventing structure.

## Pick the next model for diversity, not strength (s6e6 lesson)

A stacker rewards a model that is **wrong differently**, not one that is slightly more right.
Once 2–3 GBDTs exist, another tuned GBDT variant, seed bag, or Optuna pass on a strong base is
almost always flat at the stack (s6e6 sat at 0.9662 through all of them). What breaks a plateau
is a base that differs on a real axis:

- **different feature space** (e.g. heavy target-encoding recipe vs raw features),
- **different loss / model class** (e.g. a neural net with logit-adjusted loss instead of
  reweighting — biggest stack lift on s6e6, +0.0019),
- **different decomposition** (e.g. chain-cascade: factor multiclass into sequential binaries).

When porting a reference solution (public notebook), reproduce it **fully** before concluding
the idea is flat — partial reproductions of strong recipes score as noise (s6e6's `lgbm_fe`).

## The contract every `train_<model>.py` must honour

1. **Load through `build_features()`** — never read `train.csv`/`test.csv` and pull `y`/`id`
   directly. The `oof_*.npy` / `test_*.npy` arrays are saved in `build_features()` sort order;
   bypassing it silently misaligns predictions (see the row-order invariant — caused a 0.5-AUC
   submission once).
2. **CV**: `run_cv(X, y, X_test, fit_fold, ...)` from `data_science_stuff.kaggle.cv` — it owns
   the `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` skeleton (`stratified=False`
   → `KFold` for regression). The same `random_state=42` across all models keeps OOF folds
   aligned. Your `fit_fold(X_tr, y_tr, X_va, y_va, X_test, fold)` returns
   `(val_pred, test_pred)`; fold-aware transforms (target encoding, scalers) live INSIDE it,
   and the `after_fold` hook is where `torch.cuda.empty_cache()` goes.
3. **Outputs** written to `results/` (paths via
   `DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)` from
   `data_science_stuff.kaggle.io`; submissions via `write_submission(...)`):
   - `results/oof_<model>.npy` — out-of-fold predictions, length = n_train, in sort order.
   - `results/test_<model>.npy` — mean of per-fold test predictions, length = n_test.
   - Append a row via `save_cv_result(RESULTS_DIR, "<model>_vN", fold_scores, oof_score)` — the
     shared helper from `data_science_stuff.kaggle_utils` (re-exported by each competition's
     `src/cv_results.py`). Pass `metric_name="..."` for non-AUC metrics (e.g. `"balanced_acc"`).
   - Optionally a `submissions/<model>_vN.csv`.
4. **Exclude** the same columns as the baseline: `{"id", TARGET} | DRIVER_COLS` (or the
   competition's equivalent). Pass string categoricals as `category` dtype to LGBM/XGBoost,
   `cat_features` to CatBoost, OHE to neural nets.
5. **Multiclass / imbalanced classification** (s6e6 conventions):
   - `oof_<model>.npy` / `test_<model>.npy` store **raw probabilities** (n × n_classes), never
     argmax labels — stackers and threshold optimization need the full distribution.
   - Target via `LabelEncoder`; always `le.inverse_transform` when writing submissions.
   - Set `class_weight="balanced"` (or the library's equivalent) — biggest cheap win on
     imbalanced targets; prior correction on top of it adds nothing. For NNs, prefer
     `data_science_stuff.kaggle.models.losses.logit_adjustment` (balanced softmax) over class
     weights — it was the s6e6 rare-class fix where weights/oversampling were flat.
   - After CV, `from data_science_stuff.kaggle.decision import optimize_thresholds` →
     `(weights, score)`; save with `kaggle.io.save_threshold_weights` and apply the identical
     weights to test.
6. **Fold-aware target encoding**: use
   `data_science_stuff.kaggle.encoding.add_fold_safe_target_encoding(X_tr, y_tr, [X_va, X_test],
   te_cols, class_map)` INSIDE the fold loop — it fits per-class binary TargetEncoders on the
   fold's training split only (nested cv). Never fit on all of `(X, y)`. Quantile-bin TE
   sources come from `kaggle.encoding.add_quantile_bin_features` / `qcut_codes`.

Worked examples of package usage (import the package, don't copy these):
`playground-series-s6e6/src/baseline.py` (LGBM + run_cv + device probe),
`train_mlp.py`/`train_mlp_la.py` (kaggle.models.mlp.fit_mlp_fold, logit adjustment),
`train_xgb_deotte.py` (fold-safe TE recipe), `train_chain_cascade.py` (cascade_combine
decomposition); `playground-series-s6e5/src/train_xgboost.py` (bespoke in-loop fold transforms
where run_cv's contract doesn't fit). Reusable tabular NNs live in
`data_science_stuff.kaggle.models` (`MLP`/`fit_mlp_fold`, `RealMLP_TD_Classifier` with the
`loss_prior_power` knob).

## GPU memory rule (PyTorch models — mandatory)

Any function that allocates GPU tensors/models must `del` them before returning, and the outer
fold/trial loop must call `torch.cuda.empty_cache()` after each iteration. Omitting this OOM-crashes
multi-fold/multi-trial runs. Pattern from `train_mlp.py`:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    model = MLP(...).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), ...)
    ...  # train, collect oof/test predictions
    del model, optimizer, scheduler, X_tr_t, y_tr_t, X_val_t, X_test_t
    torch.cuda.empty_cache()   # after each fold/trial
```

This applies to both `train_*.py` and `tune_*.py`.

## Device selection gotchas (s6e6 lessons)

- **LightGBM**: pip wheels are usually CPU-only — `device_type="cuda"` raises at fit time.
  `from data_science_stuff.kaggle.device import get_lgbm_device` probes once (cached) and
  returns `("cuda", 1)` or `("cpu", -1)`; use it in both train and tune scripts so they agree
  (GPU wants `n_jobs=1`, CPU wants all cores). Never use `device="gpu"` (the OpenCL backend)
  — it is slow / silently falls back on NVIDIA; the helper probes `"cuda"` only.
- **pytabkit**: check the constructor mixin in `sklearn_interfaces.py` before setting speed
  params — mixins have different param sets:
  - `TabM_D_Classifier` (TabMConstructorMixin): `compile_model=True` + `allow_amp=True` +
    `batch_size=512` + `eval_batch_size=8192`. Forgetting these turns a ~1h fold into ~3h.
  - `RealTabR_D_Classifier` (TabrConstructorMixin): does NOT accept `compile_model`/`allow_amp`.
    Use `batch_size=512`, `freeze_contexts_after_n_epochs=3`; OOM knobs `eval_batch_size=1024`,
    `candidate_encoding_batch_size=2048`, and set
    `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` before fit.

## Optional `tune_<model>.py` (Optuna)

Before writing one: tuning an already-strong base rarely moves the *ensemble* (s6e6: XGB tuning
was a wash on the honest argmax metric, and the tuned blend matched the untuned one — stronger
bases just grow more correlated). Prefer spending the compute on a diverse new base; tune when a
model is clearly undertrained or is the only member of its family.

Mirror `playground-series-s6e5/src/tune.py` / `tune_mlp.py`:
- `objective(trial, X, y)` runs an inner `StratifiedKFold` (commonly `N_FOLDS = 3` for speed,
  `N_TRIALS = 50`) and returns the OOF metric.
- `study = optuna.create_study(direction="maximize")` (or `"minimize"` for error metrics).
- Save the best params to `results/best_params_<model>.json`; the matching `train_<model>.py`
  loads them via `load_params(RESULTS_DIR, DEFAULTS, "best_params_<model>.json")` from
  `data_science_stuff.kaggle.io` (returns the defaults when the file is absent).
- Tree models can run on GPU (`device="gpu"`/`"cuda"`, `task_type="GPU"`); apply the GPU memory
  rule to any PyTorch tuning loop.

## Style

Polars I/O → pandas at fit time; type hints everywhere (mypy strict); Python 3.9 lowercase
generics; paths via `competition_dirs(__file__)`. `sys.path.insert(0,
str(Path(__file__).parent))` is only for the competition's own modules (`features`,
`cv_results`); shared utilities are normal `data_science_stuff.kaggle` imports.

## Verify

`cd <competition> && python src/train_<model>.py` — it should print per-fold + OOF scores and
write `results/oof_<model>.npy` and `results/test_<model>.npy`. Confirm the arrays have the
expected lengths (n_train / n_test). Then run the `quality-gate` skill. After adding a model,
rerun the `ensemble-submit` skill so the blend includes it.
