# PS S6E5 — F1 Pit Stop Prediction

**Task**: Binary classification — predict `PitNextLap` (will this driver pit on the next lap?)
**Metric**: AUC-ROC
**Deadline**: May 31, 2026

---

## ⚠️ CRITICAL: Row-order invariant — always use build_features() to load labels and IDs

`features.py::build_features()` sorts every dataframe by `["Driver", "Race", "Year", "LapNumber"]`.
All `oof_{model}.npy` and `test_{model}.npy` arrays are in this sorted order.

Any script that loads `y` or `test_ids` and combines them with npy arrays **must** go through `build_features()`:

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — silently misaligns predictions → 0.5 AUC on Kaggle
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

This bug caused a 0.5 AUC submission from `ensemble.py`. Re-run `ensemble.py` before any submission.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 439,140 | 16 (incl. target) |
| test | 188,165 | 15 |

**No missing values** anywhere. Target: 80.1% zeros, 19.9% ones — stratified CV is sufficient.

### Raw features

`Driver`, `Compound`, `Race`, `Year`, `PitStop`, `LapNumber`, `Stint`, `TyreLife`, `Position`,
`LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`

---

## Key EDA Findings

### ⚠️ 2023 Data Anomaly — CRITICAL

Year 2023 has a pit rate of **0.96%** versus ~27–30% for all other years — a labelling artefact
in the synthetic data. **Never drop 2023** (OOF crashed to 0.9147). The `is_2023` flag lets
models use those 136k rows as clean negative examples while isolating the anomaly.

| Year | Pit rate | N | Train% | Test% |
|---|---|---|---|---|
| 2022 | 26.7% | 82,989 | 18.9% | 18.8% |
| 2023 | **0.96%** | 136,147 | 31.0% | 30.9% |
| 2024 | 29.5% | 127,110 | 28.9% | 29.0% |
| 2025 | 28.4% | 92,894 | 21.2% | 21.3% |

### Train / Test Distribution — No Shift (2026-05-06)

Adversarial validation AUC = **0.5013**. All 26 Races and all 104 (Race, Year) tuples
present in both splits. Race (26), Compound (5), Year (4) fully overlap — no unseen
categories at inference time. **1,364 (Driver, Race, Year) tuples are test-only** — these
novel stints are where generalisation matters most.

KS statistics for all numeric features < 0.004. Full analysis: `notebooks/distribution_shift.py`.

### Key predictors (SHAP, LGBM v5)

1. `is_2023` (0.794) — dominates; model spends most capacity on the anomaly
2. `Stint` (0.576)
3. `LapTime_Delta` (0.396) — beats raw TyreLife
4. `TyreLife` (0.369)
5. `Year` (0.342) — partly a proxy for 2023
6. `race_progress_x_stint` (0.203), `TyreLife_sq` (0.202), `race_compound_pit_rate` (0.200)

### Other EDA notes
- `TyreLife` strongest raw correlate (+0.274): pitting laps have mean TyreLife 19.5 vs 12.8 non-pitting
- `Cumulative_Degradation` has counterintuitive −0.167 correlation — verify whether it resets on pit
- HARD compound has highest pit rate (32.8%) due to right-shifted TyreLife distribution, not tyre quality
- `PitStop` (this lap) slightly elevated next-lap pit rate (24.8% vs 19.1%) — valid sequential feature, not leakage
- Near-perfect calibration across all bins; no value in post-hoc calibration

---

## AUC by Year — CRITICAL: headline OOF is inflated

| Year | N | pit% | LGBM | XGB | CatBoost | MLP | Ensemble |
|---|---|---|---|---|---|---|---|
| 2022 | 82,989 | 26.7% | 0.9090 | 0.9112 | 0.9109 | 0.9086 | 0.9139 |
| 2023 | 136,147 | 1.0% | 0.9383 | 0.9428 | 0.9335 | 0.9126 | 0.9422 |
| 2024 | 127,110 | 29.5% | 0.9244 | 0.9269 | 0.9255 | 0.9234 | 0.9284 |
| 2025 | 92,894 | 28.4% | 0.9239 | 0.9261 | 0.9248 | 0.9220 | 0.9275 |
| **Overall** | 439,140 | 19.9% | 0.9474 | 0.9489 | 0.9479 | 0.9457 | 0.9500 |

OOF inflation ~+0.027 across all models. **Expected LB ≈ 0.929 (ensemble).**

- MLP on 2023: AUC 0.9126 vs trees 0.9335–0.9428 — MLP far weaker at exploiting the easy anomaly
- Year 2022 is hardest — first season of major regulation changes, most strategy variance
- Fresh tyres (1–3 laps): AUC 0.8918 — hardest regime (undercuts, safety cars)
- Old tyres (28–77 laps): AUC 0.9509 — obvious necessity
- WET compound: AUC 0.8620 (n=1355 — too small to learn reliably)

### Next steps with highest expected value
1. **Conditional ensemble (is_2023 split)**: route 2023 rows to trees only (AUC 0.9335–0.9428 vs MLP 0.9126)
2. **MLP Optuna re-tune** on OHE feature set (67 features) — current params tuned on old 38-feature setup
3. **Autoregressive "overdue" feature**: cumulative OOF pit probability within current stint — captures strategic suppression not in TyreLife alone; requires two model passes + sequential test inference

---

## Current Best

**Ensemble OOF: 0.9508** (2026-05-09) ← current best
Conditional blend (5 models, lgbm_ar added): 2023→LGBM 43.6%+LGBM-AR 17.9%+XGB 31.9%+MLP 6.6%; non-2023→LGBM 44.4%+LGBM-AR 25.3%+MLP 28.5%+CB 1.7%

Previous best: **0.9507** (2026-05-09) 4-model conditional blend

| Model | OOF AUC | Script |
|---|---|---|
| LGBM | 0.9500 | `src/baseline.py` (lgbm_v11) |
| LGBM-AR | 0.9498 | `src/train_lgbm_ar.py` (lgbm_ar_v1) — autoregressive overdue feature |
| XGBoost | 0.9494 | `src/train_xgboost.py` (xgb_v4) |
| CatBoost | 0.9473 | `src/train_catboost.py` (catboost_v4) |
| MLP | 0.9461 | `src/train_mlp.py` (mlp_v7) |

---

## Feature Engineering

All features implemented in `src/features.py`. Currently active:

- **v1**: `is_2023`, `TyreLife_sq`
- **v2**: `TyreLife_log`, `compound_ord`, `tyre_life_x_compound`, `race_progress_x_stint`, `degradation_rate`, `est_laps_remaining`
- **rolling**: `lap_time_delta_roll3/7`, `lap_time_s_roll5`, `lap_time_delta_lag1/2/3`, `lap_time_vs_roll5`, `pace_anomaly_pct`
- **group stats** (train-only, joined to test): `driver_pit_rate`, `driver_compound_pit_rate`, `race_compound_pit_rate`, `driver/race_compound_median_tyre_life_at_pit`, `TyreLife_frac`, `window_gap`, `laps_past_window`

### What didn't help (LGBM baseline)
| Idea | Delta | Why |
|---|---|---|
| Polynomial / interaction transforms (v2) | −0.0001 | LGBM finds these internally |
| Group aggregates on LGBM (v3) | −0.0002 | Smooth rates add noise at LGBM level |
| Sequential features lag/roll (v6) | −0.0003 | Deep trees already capture this |
| `TyreLife_frac` | −0.0002 | Redundant with existing tyre features |
| Drop 2023 entirely (v4) | −0.029 | Removes 136k clean negatives |
| `degradation_rate_pace` + `lap_time_delta_ewma5` | 0.0000 | LGBM captures LapTime_Delta×TyreLife interactions and recency weighting internally |

---

## Driver Features

`DRIVER_COLS = frozenset({"Driver"})` — only the raw Driver string categorical is excluded.
Numeric driver stats (`driver_pit_rate`, `driver_compound_pit_rate`, `driver_median_tyre_life_at_pit`) are kept.

**Key insight**: the raw Driver categorical (887 values, 86 train-only drivers) hurts LGBM
via memorisation. The numeric aggregates are computable on test and are universally helpful.
Dropping all driver features (tried) left ensemble flat; dropping only the string gave +0.0006.
CatBoost with Driver restored: solo +0.0005 (0.9478) but ensemble −0.0001 — Driver gains in CB
displace XGB weight in the blend, netting a loss. Driver stays excluded for all models.

| Model | With Driver | Drop Driver string only | Drop all driver |
|---|---|---|---|
| LGBM | 0.9474 | **0.9500** | 0.9494 |
| XGBoost | 0.9489 | 0.9493 | 0.9486 |
| CatBoost | 0.9479 | 0.9476 | 0.9471 |
| MLP | 0.9457 | 0.9458 | 0.9459 |
| **Ensemble** | **0.9500** | **0.9506** | 0.9500 |

---

## Modelling Notes

- **CV**: 5-fold stratified, `random_state=42`. Scores logged to `results/cv_scores.csv`
- **LGBM**: GPU (`device="gpu"`) — previously CPU-only due to Driver's 887 unique values exceeding the GPU 256-bin limit; switched to GPU after Driver string was dropped
- **XGBoost / CatBoost**: `device="cuda"` / `task_type="GPU"` — no bin restriction
- **Categoricals**: `Compound` + `Race` passed as `category` dtype to LGBM/XGBoost, `cat_features` to CatBoost, OHE (`pd.get_dummies`) to MLP
- **MLP preprocessing**: Yeo-Johnson on numeric cols, StandardScaler on OHE binary cols
- **GPU memory rule (all MLP scripts)**: `del` model + optimizer + tensors before returning from each fold function; call `torch.cuda.empty_cache()` after each fold/trial in the outer loop. Applies to both `train_mlp.py` and `tune_mlp.py`. Omitting this in a 150-iteration tuning run (50 trials × 3 folds) crashed the PC
- **MLP params** (Optuna, 50 trials): 4-layer [1024→683→456→304], dropout=0.35, lr=4.8e-3, wd=1.9e-3 — tuned on 38-feature setup, needs re-tune for 67-feature OHE

---

## Next Steps (remove each item when implemented)

Priority order: E → C

**C. HPO re-run** — all models were tuned with Driver in feature set; params are now stale.
MLP is most critical (38→67 features via OHE). Run `tune_mlp.py` first, then
`tune.py` / `tune_xgboost.py` / `tune_catboost.py` as time permits.

**E. is_2022 flag** — add `(pl.col("Year") == 2022).cast(pl.Int8).alias("is_2022")` to
`features.py::build_features()`. Year 2022 ensemble AUC is 0.9139 vs 0.9284+ for
2024/2025; mirrors `is_2023` logic. One line, test in LGBM first.

---

## Post-Ensemble Submission Blending

Strategies from public competition notebooks — untried by us; use as inspiration for the blending phase once model training plateaus.

### Rank blending
Instead of linearly mixing probabilities, blend the *ranks* of two submissions, then re-assign the anchor's sorted probability values to the new ordering. Preserves the anchor's calibration (distribution shape unchanged) while potentially improving AUC-relevant ordering.

```python
def rank_blend(anchor, support, w):
    anchor_rank = np.argsort(np.argsort(anchor, kind="mergesort"), kind="mergesort") / len(anchor)
    support_rank = np.argsort(np.argsort(support, kind="mergesort"), kind="mergesort") / len(support)
    blended_rank = (1 - w) * anchor_rank + w * support_rank
    order = np.argsort(blended_rank, kind="mergesort")
    out = np.empty_like(anchor)
    out[order] = np.sort(anchor)
    return out
```

Try `w=0.02` (2%) with the next most diverse strong submission as support.

### Selective consensus correction
Build a consensus from N support submissions (e.g. `0.40*s_a + 0.35*s_b + 0.25*s_c`), then only modify the rows where `|consensus − anchor|` is largest (top 2%). Apply a mild blend weight (≈10% consensus) only on those rows. Targets correction effort at rows most likely to be wrong without diluting the whole submission.

Useful variants:
- **Up-only**: correct only rows where `consensus > anchor` — tests for systematic under-estimation
- **Down-only**: correct only rows where `consensus < anchor` — tests for systematic over-estimation
- **Extreme-rank**: restrict to top/bottom 10% of anchor predictions (where AUC is most sensitive) before selecting the top-2% disagreement rows

### Blend exhaustion signal
When every support submission has `corr > 0.999` and `mean_abs_delta < 0.001` to the anchor, linear micro-blends will not move the leaderboard — the signal set is exhausted. The next gain requires a genuinely *different* high-scoring submission, not tighter mixing of existing ones. Track `corr` and `mean_abs_delta` to anchor when evaluating new blend candidates.

**Tested 2026-05-09**: Support submissions corr range 0.965–0.998 (diverse by definition), yet best blend delta = 5e-6 (noise). The RNN (most diverse at corr=0.965) has too low quality (OOF 0.9368) to contribute positively; the better models (corr 0.997–0.998) lack the ordering diversity needed. Signal is exhausted — the next gain requires a structurally different high-quality model, not blending of the current set.

### Submission diversity taxonomy
Before blending, classify submissions by tier using `corr + mean_abs_delta` to the anchor:

| Tier | Description | Expected corr to anchor |
|---|---|---|
| super | Current top-LB cluster | > 0.9998 |
| core | Strong cluster 0.001–0.003 below top | ~0.9996 |
| diverse | Structurally different (e.g. mean ≈ 0.5) | < 0.998 |
| support | Older/lower-scoring but complementary | ~0.997–0.999 |
| top_external | Other teams' public best submissions | varies |

Diversity (low corr / high delta) is a prerequisite for blending to help — two submissions that are 0.9999 correlated will not correct each other's errors.

---

## Experiments Log

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-05-04 | lgbm_v1 | Raw features + `is_2023` + `TyreLife_sq` | 0.9433 |
| 2026-05-04 | lgbm_v2–v3 | +transforms, +group aggregates | 0.9431–0.9432 (flat) |
| 2026-05-04 | lgbm_v4 | Drop 2023 | 0.9147 (−0.029) |
| 2026-05-05 | lgbm_v5 | Optuna-tuned (num_leaves=490, lr=0.023) | **0.9480** (+0.0047) |
| 2026-05-05 | lgbm_v6/v8 | +sequential features | 0.9474–0.9477 (flat) |
| 2026-05-05 | catboost_v1 | GPU defaults | 0.9471 |
| 2026-05-05 | catboost_v1 (tuned) | Optuna: depth=9, lr=0.060 | 0.9479 |
| 2026-05-05 | xgboost_v1 | GPU defaults | 0.9456 |
| 2026-05-05 | xgboost_v1 (tuned) | Optuna: best trial 33 | **0.9489** (+0.0033) |
| 2026-05-05 | ensemble (3-model tuned) | XGB 50.4% + CB 31.6% + LGBM 17.9% | **0.9495** |
| 2026-05-05 | mlp_v1 | PyTorch GPU, 512→256→128 | 0.9448 |
| 2026-05-05 | mlp_v2 | +Yeo-Johnson preprocessing | 0.9452 |
| 2026-05-05 | mlp_v3 | +Optuna tune, 4-layer [1024→683→456→304] | 0.9457 |
| 2026-05-05 | ensemble (4-model, mlp_v3) | XGB 46.4% + MLP 26.6% + CB 18.1% + LGBM 8.9% | **0.9500** |
| 2026-05-05 | stacking_v1 | LR meta-learner, nested CV | 0.9500 (flat) |
| 2026-05-06 | all models (drop Driver string only) | lgbm_v10, xgb_v3, catboost_v3, mlp_v5 | see Driver table |
| 2026-05-06 | ensemble w/ mlp_v5 | LGBM 52.7% + XGB 23.9% + MLP 23.4% + CB 0% | **0.9506** (new best) |
| 2026-05-06 | mlp_v6 | +OHE for Compound/Race (38→67 features); params not re-tuned | 0.9455 (−0.0003) |
| 2026-05-06 | ensemble w/ mlp_v6 | same weights | 0.9506 (flat) |
| 2026-05-08 | all models re-tuned (GPU for LGBM) | lgbm_v11, xgb_v4, catboost_v4, mlp_v7 — re-tuned params post Driver-drop | LGBM 0.9500, XGB 0.9494, CB 0.9473, MLP 0.9461 |
| 2026-05-08 | ensemble w/ all v4/v7 | LGBM 65.8% + MLP 25.2% + XGB 9.0% + CB 0% | **0.9506** (flat) |
| 2026-05-09 | stacking_v2_lr_with_features | LR meta-learner (L2, C=0.1/1/10) + full feature matrix alongside OOF logits | 0.9506 (flat — LR learns nothing extra from raw features on top of calibrated OOFs) |
| 2026-05-09 | rnn_v1 | GRU hidden=256 layers=2 dropout=0.3, GroupKFold(5), 50 epochs — src/train_rnn.py | 0.9368 (below MLP 0.9461; no tuning; folds balanced so gap is real — needs Optuna before ensemble use) |
| 2026-05-09 | ensemble_v3 | Conditional blend split on is_2023 — separate Nelder-Mead per regime | **0.9507** (+0.0001; 2023→LGBM 57.5%+XGB 35.5%+MLP 7%; non-2023→LGBM 66.4%+MLP 30%) |
| 2026-05-09 | lgbm_ar_v1 | Autoregressive overdue feature (cumsum OOF within stint, lag 1); sequential test inference — src/train_lgbm_ar.py | 0.9498 (−0.0002 vs lgbm; adds diversity via different structure) |
| 2026-05-09 | ensemble_v4 | Conditional blend with 5 models (lgbm_ar added); 2023→LGBM 43.6%+LGBM-AR 17.9%+XGB 31.9%+MLP 6.6%; non-2023→LGBM 44.4%+LGBM-AR 25.3%+MLP 28.5% | **0.9508** (new best; +0.0001) |
| 2026-05-09 | pace features (item D) | `degradation_rate_pace` + `lap_time_delta_ewma5` — reverted; OOF flat at 0.9500; LGBM already captures these interactions internally | 0.9500 (no change) |
| 2026-05-09 | catboost_v5 (item B) | Driver string restored as cat feature — reverted; solo +0.0005 (0.9478) but ensemble −0.0001 (0.9507); Driver displacement of XGBoost weight harms blend | 0.9478 solo / 0.9507 ensemble |
| 2026-05-09 | blend_submissions (rank+selective) | Post-ensemble rank blend + selective consensus correction; all support submissions diverse (corr 0.96–0.998) but best blend delta = 5e-6 (noise); signal exhausted with current model set | 0.9508 (flat) |
