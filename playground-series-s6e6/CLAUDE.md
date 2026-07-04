# PS S6E6 — Predicting Stellar Class

**Task**: 3-class classification — predict `class` ∈ {GALAXY, STAR, QSO}
**Metric**: Balanced accuracy
**Deadline**: June 30, 2026

---

## ⚠️ NO DATA LEAKAGE — mandatory rules

These rules must be followed in every script. Violating them produces inflated OOF scores that do not generalise to the leaderboard.

### Rule 1 — Group stats use train_raw, never a fold slice

`compute_group_features(train_raw, df)` always receives the **full raw training CSV** as its reference. This is intentional: these are global priors, not fold-specific stats. They are low-leakage because they aggregate feature columns across all rows, not per-row targets.

### Rule 2 — Target encoding is fold-aware (NEVER global)

Any `TargetEncoder` or smoothed target-rate feature MUST be fit **only on the fold's training split**, then transformed on the val split:

```python
# CORRECT
te = TargetEncoder(smooth="auto", cv=5, random_state=42)
X_tr["enc"] = te.fit_transform(X_tr[["cat_col"]], y_tr).ravel()
X_val["enc"] = te.transform(X_val[["cat_col"]]).ravel()

# WRONG — leaks val targets into encoding
te.fit(X[["cat_col"]], y)
```

### Rule 3 — Test features must be computable without the target

`build_features(test)` must never depend on the test target (which does not exist at inference time). `compute_group_features(train_raw, test)` is fine — it uses only train targets. Sequential/lag features on test must shift by ≥1 row.

### Rule 4 — Row-order invariant

`build_features()` sorts every frame by `SORT_KEY`. All `oof_*.npy` and `test_*.npy` arrays are in this sorted order. Any script that loads `y` or `test_ids` and combines them with `.npy` arrays **must** go through `build_features()`:

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — silently misaligns predictions → random leaderboard score
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

---

## DRY Code — mandatory conventions

- **`features.py` is the single source of truth** for all feature engineering. No training script may define its own transforms or constants (feature lists, column exclusions, sort keys).
- **Shared utilities live in `data_science_stuff.kaggle`** (io / device / cv / blending / decision / encoding / stacking / models) — import, never copy. `postprocess.py`, `lgbm_device.py`, and `realmlp_deotte.py` are thin re-export shims onto the package; `cv_results.py` re-exports `kaggle_utils.save_cv_result`. Extend the package, not the shims.
- The shared CV loop is `data_science_stuff.kaggle.cv.run_cv` (adopted by baseline / train_xgboost / train_catboost / train_mlp / train_mlp_la); long-tail experiment scripts with bespoke fold logic keep their own loops.
- Categorical exclusion sets, sort keys, and the target name are constants in `features.py` — imported everywhere, never redefined.
- **Refactored onto `data_science_stuff.kaggle` (2026-07-04).** All historical scores in this file were produced by the pre-refactor scripts. The threshold optimizer defaults changed with the extraction (max-normalized weights — invariant under `argmax(p·w)`; class-0-fixed parameterization; different restart RNG) — decision-rule-equivalent, not bit-identical. `results/` artifacts on disk are untouched.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 577,347 | 12 (10 features + id + class) |
| test | 247,435 | 11 (10 features + id) |

**Target classes**: GALAXY (65.4%), QSO (20.3%), STAR (14.3%) — moderately imbalanced
**Missing values**: none

### Raw features

| Feature | Type | Notes |
|---|---|---|
| `alpha` | float | Right ascension (sky coordinate) |
| `delta` | float | Declination (sky coordinate) |
| `u`, `g`, `r`, `i`, `z` | float | SDSS photometric band magnitudes |
| `redshift` | float | Range −0.01 to 7.01; median 0.50; small negatives are measurement noise |
| `spectral_type` | cat (4) | O/B, A/F, G/K, M (Morgan-Keenan classification) |
| `galaxy_population` | cat (2) | Red_Sequence, Blue_Cloud |

---

## Key EDA Findings

- **`redshift` is the single most discriminating feature**: STAR ≈ 0.07, GALAXY ≈ 0.51, QSO ≈ 1.88 (means)
- **Photometric colors separate classes**: GALAXY has much higher u−g (1.72) vs QSO (0.56) vs STAR (1.37)
- `spectral_type` and `galaxy_population` are strong categoricals — LGBM handles them natively
- No missing values anywhere
- Small negative redshifts (min −0.01) are measurement noise, not anomalies

---

## Feature Engineering

All features implemented in `src/features.py`. Currently active: u_g, g_r, r_i, i_z, u_z color indices + log1p_redshift.

### What didn't help

| Idea | Delta | Why |
|---|---|---|
| SDSS color indices (u−g, g−r, r−i, i−z, u−z) + log1p(redshift) | +0.000 | LGBM learns linear combinations of raw bands natively; redundant |

---

## Modelling Notes

- **CV**: 5-fold stratified, `random_state=42`. Scores logged to `results/cv_scores.csv`.
- **Metric**: `balanced_accuracy_score` from sklearn (accounts for class imbalance).
- **LGBM**: `objective="multiclass"`, `num_class=3`, `metric="multi_logloss"`. **Device is auto-detected** via `src/lgbm_device.py::get_lgbm_device()` — imported by both `baseline.py` and `tune_lgbm.py` so the tuning search and final retrain always agree. The pip `lightgbm` 4.6 wheel in this `.venv` is **CPU-only** (not built with `-DUSE_CUDA=1`), so `device_type="cuda"` raises `LightGBMError: CUDA Tree Learner was not enabled in this build` at fit time. The helper probes for a working CUDA build once and returns `("cpu", -1)` here (use all cores) — for only 16 features the CUDA tree learner buys little over multi-core CPU anyway. If a CUDA-enabled lightgbm is ever installed, the helper returns `("cuda", 1)` automatically (GPU wants `n_jobs=1`; CPU threads add overhead). Note: `device="gpu"` (the OpenCL backend) is slow/silent-fallback on NVIDIA — never use it; the helper only probes `"cuda"`.
- **Label encoding**: `sklearn.preprocessing.LabelEncoder` — always use `le.inverse_transform` when writing submissions.
- **OOF arrays**: `oof_lgbm.npy` stores raw probabilities (shape: n_train × 3) for ensemble blending. `test_lgbm.npy` stores raw probabilities (shape: n_test × 3). All models follow this convention.
- **class_weight="balanced"**: Fixed in all LGBM runs — lifts STAR recall (the bottleneck class at 14% frequency) and is worth +0.008 balanced_acc over the uniform default.
- **Prior correction** (`argmax(P / train_prior)`): Adds nothing when `class_weight="balanced"` is already set; skip it.
- **Threshold weight optimisation**: After CV, optimise per-class scale factors (`argmax(proba * w)`) on OOF with Nelder-Mead (15 random restarts). Worth +0.0013 balanced_acc. Weights saved to `results/threshold_weights_*.json`. Apply identical weights to test predictions. Ensemble optimises its own threshold weights on blended OOF after blend-weight search.
- **GPU memory rule** (if MLP added): `del` model + optimizer + tensors before returning from each fold function; call `torch.cuda.empty_cache()` after each fold/trial in the outer loop.
- **pytabkit GPU speed:**
  - **`TabM_D_Classifier`** (TabMConstructorMixin): use `compile_model=True` (Inductor JIT ~10-30%) + `allow_amp=True` (fp16 Tensor Cores ~2x) + `batch_size=512` + `eval_batch_size=8192`. Forgetting these turns a ~1h fold into ~3h.
  - **`RealTabR_D_Classifier`** (TabrConstructorMixin): does NOT support `compile_model`/`allow_amp` — different mixin. Speed levers: `batch_size=512`, `freeze_contexts_after_n_epochs=3` (stops re-encoding 577k candidates every step after 3 warm-up epochs — the main compute bottleneck). **OOM knobs**: `eval_batch_size=1024` (NOT 4096 — val allocates eval_bs×96×265 dims at once → OOM at 4096+), `candidate_encoding_batch_size=2048`. Also set `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` before fit to reduce fragmentation.
  - Check the mixin class in `sklearn_interfaces.py` before adding speed params — `TabMConstructorMixin` and `TabrConstructorMixin` have different param sets.

---

## Current Best

**0.9698 / 0.9699** OOF balanced_acc — LR stacker (`build_lr_stack.py`) over 12 bases incl. two
faithful local ports of cdeotte's notebooks: `xgb_deotte` (240-feature recipe w/ fold-safe qbin
target-encoding, OOF 0.96699) and `realmlp_deotte` (from-scratch RealMLP w/ PBLD embeddings +
logit-adjust loss, OOF 0.96888). Submission `submissions/lrstack.csv`. Matches cdeotte's 0.9702
stacker with 2 of his 3 bases ported (CatBoost 0.9697 not yet ported).

Earlier plateau was **0.9662** (stuck for a full session); the break came from porting his STRONG
bases — not from tuning/blending our own ~0.965 cluster. Lesson: feed the stacker strong DIVERSE
bases (the non-GBDT RealMLP gave +0.0019); reproduce reference solutions FULLY before concluding a
lever is flat (our partial FE reproductions `lgbm_fe`/`fe2` were flat only because incomplete).

---

## Experiments Log

| Date | Run | Description | OOF balanced_acc |
|---|---|---|---|
| 2026-06-01 | baseline_lgbm_v1 | LGBM raw features only | 0.9559 |
| 2026-06-01 | lgbm_v2_colors | + SDSS color indices + log1p_redshift | 0.9559 |
| 2026-06-03 | lgbm_v3_balanced | + class_weight="balanced"; OOF now proba | 0.9638 |
| 2026-06-03 | lgbm_v3_balanced + threshold | + per-class threshold weight optimisation | 0.9651 |
| 2026-06-04 | lgbm_v4_tuned | Optuna 50-trial 3-fold search (CPU); argmax 0.9654 | 0.9654 |
| 2026-06-04 | lgbm_v4_tuned + threshold | + per-class threshold weight optimisation | **0.9657** |
| 2026-06-04 | xgb_v1 | XGBoost untuned (GPU), inv-freq weights + threshold | 0.9646 |
| 2026-06-04 | catboost_v1 | CatBoost untuned (GPU), auto_class_weights + threshold | 0.9609 |
| 2026-06-04 | ensemble (untuned xgb/cb) | Nelder-Mead blend (w: 0.72/0.21/0.07) + threshold | 0.9657 |
| 2026-06-04 | xgb_v2 (tuned) | Optuna 50-trial (GPU) + threshold | 0.9652 |
| 2026-06-04 | catboost_v2 (tuned) | Optuna 50-trial (GPU) + threshold | 0.9627 |
| 2026-06-04 | ensemble (tuned xgb/cb) | Nelder-Mead blend (w: 0.47/0.25/0.28) + threshold | 0.9656 |

**Tuning xgb/catboost rebalanced the blend (0.72/0.21/0.07 → 0.47/0.25/0.28) but did NOT lift the ensemble** (0.9657 → 0.9656, a 0.0001 move = threshold noise). On the comparable 5-fold *argmax* metric: catboost tuning was real (0.9609→0.9627), **xgb tuning was a wash (0.9645→0.9645)** — its "+0.0006" was only the in-sample threshold number. *Hypothesis* (not proven on a 0.0001 delta): stronger base models grew more correlated, leaving less diversity for the blend. Practical takeaway: everything has plateaued ~0.965–0.966. **To beat it, need a genuinely *different* model family (MLP/NN on standardised numerics), not more GBDT tuning.**

**Best submission**: ensemble and lgbm_v4_tuned tie within CV noise. Select **both** as the two Kaggle finals; if forced to one, prefer the **ensemble** (safer private-LB bet at a tie — averages out per-model error).

---

## Next Steps

1. ~~Download data and run EDA~~ ✓
2. ~~Run `src/baseline.py`~~ ✓ → **0.9559**
3. ~~`class_weight="balanced"`~~ ✓ → **0.9638** (+0.008)
4. ~~Threshold weight optimisation~~ ✓ → **0.9651** (+0.0013)
5. Optuna tuning (`src/tune_lgbm.py`) — running; will re-run `src/baseline.py` after
6. Add XGBoost / CatBoost training scripts (`src/train_xgboost.py`, `src/train_catboost.py`)
7. Ensemble via Nelder-Mead weight optimisation on OOF probabilities + threshold tuning
