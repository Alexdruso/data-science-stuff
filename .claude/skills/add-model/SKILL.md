---
name: add-model
description: Add a new model to a Kaggle competition in this monorepo — a train_<model>.py (and optional tune_<model>.py) that follows the repo's CV/OOF conventions and GPU memory rule. Use when asked to add or train a new model, e.g. "add a CatBoost model to s6e5", "write an XGBoost trainer", "add a tuning script for the MLP".
---

# Add a model to a competition

Use this to add a new base learner to an existing competition (e.g. `playground-series-s6e5/`).
Every model must plug into the ensemble, so the OOF/test array contract and CV setup are
non-negotiable. Mirror the existing reference scripts rather than inventing structure.

## The contract every `train_<model>.py` must honour

1. **Load through `build_features()`** — never read `train.csv`/`test.csv` and pull `y`/`id`
   directly. The `oof_*.npy` / `test_*.npy` arrays are saved in `build_features()` sort order;
   bypassing it silently misaligns predictions (see the row-order invariant — caused a 0.5-AUC
   submission once).
2. **CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for classification
   (`KFold` for regression). The same `random_state=42` across all models keeps OOF folds aligned.
3. **Outputs** written to `results/`:
   - `results/oof_<model>.npy` — out-of-fold predictions, length = n_train, in sort order.
   - `results/test_<model>.npy` — mean of per-fold test predictions, length = n_test.
   - Append a row via `save_cv_result(RESULTS_DIR, "<model>_vN", fold_scores, oof_score)` — the
     shared helper from `data_science_stuff.kaggle_utils` (re-exported by each competition's
     `src/cv_results.py`). Pass `metric_name="..."` for non-AUC metrics (e.g. `"balanced_acc"`).
   - Optionally a `submissions/<model>_vN.csv`.
4. **Exclude** the same columns as the baseline: `{"id", TARGET} | DRIVER_COLS` (or the
   competition's equivalent). Pass string categoricals as `category` dtype to LGBM/XGBoost,
   `cat_features` to CatBoost, OHE to neural nets.

Reference implementations to copy from:
`playground-series-s6e5/src/train_xgboost.py`, `train_catboost.py`, `train_mlp.py`,
`baseline.py` (LGBM), `cv_results.py`.

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

## Optional `tune_<model>.py` (Optuna)

Mirror `playground-series-s6e5/src/tune.py` / `tune_mlp.py`:
- `objective(trial, X, y)` runs an inner `StratifiedKFold` (commonly `N_FOLDS = 3` for speed,
  `N_TRIALS = 50`) and returns the OOF metric.
- `study = optuna.create_study(direction="maximize")` (or `"minimize"` for error metrics).
- Save the best params to `results/best_params_<model>.json`; the matching `train_<model>.py`
  loads them via a `load_params()` that falls back to sane defaults when the file is absent.
- Tree models can run on GPU (`device="gpu"`/`"cuda"`, `task_type="GPU"`); apply the GPU memory
  rule to any PyTorch tuning loop.

## Style

Polars I/O → pandas at fit time; type hints everywhere (mypy strict); Python 3.9 lowercase
generics; `Path(__file__).parent.parent / "data"` path pattern; cross-module imports via
`sys.path.insert(0, str(Path(__file__).parent))`.

## Verify

`cd <competition> && python src/train_<model>.py` — it should print per-fold + OOF scores and
write `results/oof_<model>.npy` and `results/test_<model>.npy`. Confirm the arrays have the
expected lengths (n_train / n_test). Then run the `quality-gate` skill. After adding a model,
rerun the `ensemble-submit` skill so the blend includes it.
