# PS S6E5 — F1 Pit Stop Prediction

**Task**: Binary classification — predict `PitNextLap` (will this driver pit on the next lap?)
**Metric**: AUC-ROC
**Deadline**: May 31, 2026

> **Keep this file up to date.** Append new findings from experiments, feature
> engineering, or model runs under the relevant section. This is the single
> source of truth for what we know about this competition.

## ⚠️ CRITICAL: Row-order invariant — always use build_features() to load labels and IDs

`features.py::build_features()` sorts every dataframe by `["Driver", "Race", "Year", "LapNumber"]`.
All model scripts call `build_features()` before generating `oof_{model}.npy` and
`test_{model}.npy`, so those arrays are in **sorted order**, not original CSV order.

**Rule**: any script that loads `y` (train labels) or `test_ids` and compares/combines
them with npy arrays **must** go through `build_features()`:

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — silently misaligns predictions → 0.5 AUC on Kaggle
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
test_ids = pl.read_csv(DATA_DIR / "test.csv")["id"].to_numpy()
```

This bug caused a 0.5 AUC submission from `ensemble.py` and was also found in
`stacking.py`. Individual model submissions were never affected because they extract
IDs from their already-sorted feature-engineered frames.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 439,140 | 16 (incl. target) |
| test | 188,165 | 15 |

**No missing values** anywhere in train or test.

### Features

| Feature | Type | Notes |
|---|---|---|
| `Driver` | categorical | Anonymised driver codes (e.g. D109, VER) |
| `Compound` | categorical | HARD / MEDIUM / SOFT / INTERMEDIATE / WET |
| `Race` | categorical | Grand Prix name |
| `Year` | int | 2022–2025 — **see anomaly note below** |
| `PitStop` | binary int | Did the driver pit *this* lap |
| `LapNumber` | int | Lap within the race |
| `Stint` | int | Current tyre stint number |
| `TyreLife` | float | Age of current tyre in laps |
| `Position` | int | Current race position |
| `LapTime (s)` | float | Lap time in seconds |
| `LapTime_Delta` | float | Change in lap time vs previous lap |
| `Cumulative_Degradation` | float | Cumulative tyre degradation signal |
| `RaceProgress` | float | Fraction of race completed (0–1) |
| `Position_Change` | float | Change in position vs previous lap |

---

## Target Distribution

- `PitNextLap = 0`: 351,759 (80.1%)
- `PitNextLap = 1`: 87,381 (19.9%)

Mild imbalance — stratified CV is sufficient; no need for aggressive resampling.

---

## Key EDA Findings

### ⚠️ 2023 Data Anomaly — CRITICAL

Year 2023 has a pit rate of **0.96%** versus ~27–30% for 2022, 2024, 2025. This is
almost certainly a labelling or generation artefact in the synthetic data, not a real
F1 strategy shift. Options:
- Drop 2023 rows from training entirely
- Flag 2023 as a binary feature and investigate its effect on CV
- Never trust year-based pit rate statistics that include 2023 uncorrected

| Year | Pit rate | N |
|---|---|---|
| 2022 | 26.7% | 82,989 |
| 2023 | **0.96%** | 136,147 |
| 2024 | 29.5% | 127,110 |
| 2025 | 28.4% | 92,894 |

### Top Feature Correlations with Target

| Feature | Pearson r |
|---|---|
| `TyreLife` | **+0.274** |
| `LapNumber` | +0.267 |
| `Stint` | +0.198 |
| `RaceProgress` | +0.185 |
| `Cumulative_Degradation` | −0.167 |
| `PitStop` (this lap) | +0.049 |
| `Position_Change` | +0.046 |
| `LapTime (s)` | −0.034 |
| `Position` | +0.021 |
| `LapTime_Delta` | −0.005 |

### TyreLife — Strongest Individual Predictor

Pitting drivers have substantially older tyres:

| PitNextLap | Mean TyreLife | Median | P90 |
|---|---|---|---|
| 0 | 12.8 | 11.0 | 26.0 |
| 1 | 19.5 | 18.0 | 33.0 |

### Compound Pit Rates — Counterintuitive Ordering

HARD has the highest pit rate (32.7%). This is likely a synthetic data artefact or
reflects that HARD stints run longer and so the *distribution* of TyreLife is
right-shifted, not that HARD tyres are "worse".

| Compound | Pit rate | N |
|---|---|---|
| HARD | 32.8% | 170,518 |
| SOFT | 19.3% | 38,744 |
| INTERMEDIATE | 15.2% | 17,382 |
| MEDIUM | 10.1% | 211,141 |
| WET | 2.5% | 1,355 |

Always condition compound analysis on TyreLife to avoid confounding.

### PitStop Flag (This Lap)

Drivers who pitted *this* lap have a slightly elevated next-lap pit rate (24.8% vs
19.1%). Effect is small — not a leakage concern, valid sequential strategy feature.

### RaceProgress

Pitting laps cluster later in the race (mean 43% vs 31% for non-pitting). Reflects
typical 1–2 stop strategies. Consider interaction with Stint.

### Cumulative_Degradation — Negative Correlation

Negative correlation (−0.167) with the target is counterintuitive. Investigate
whether this feature resets on a pit stop or is computed differently to TyreLife.
Do not assume it is a direct proxy for tyre wear without verifying.

### Train / Test Distribution — Confirmed No Shift (2026-05-06)

Adversarial validation AUC = **0.5013** (5-fold LGBM, all features) — distributions
are statistically indistinguishable. Year proportions match to <0.1pp:

| Year | Train% | Test% |
|---|---|---|
| 2022 | 18.9% | 18.8% |
| 2023 | 31.0% | 30.9% |
| 2024 | 28.9% | 29.0% |
| 2025 | 21.2% | 21.3% |

All 26 Races and all 104 (Race, Year) tuples present in both splits — no held-out
race contexts. 86 drivers are train-only, 0 are test-only. However, **1,364 (Driver,
Race, Year) tuples are test-only** (specific driver appearances at races not seen in
that year's training data). These novel stints are where generalisation matters most.

KS statistics for numeric features: all < 0.004, all p-values > 0.12 — no meaningful
marginal shift. Full analysis in `notebooks/distribution_shift.py`.

---

## Feature Engineering Ideas

- `TyreLife × Compound` interaction (SOFT degrades faster per lap)
- `TyreLife² ` or log-TyreLife (non-linear degradation curve)
- `RaceProgress × Stint` (late-race stint flags strategic windows)
- Binary `is_2023` flag to isolate the anomaly year
- `LapTime_Delta` rolling mean (smooth out noise)
- `Cumulative_Degradation / TyreLife` ratio (degradation rate)

### Future: autoregressive "overdue" feature

**Idea**: Add a feature = cumulative pit probability predicted by a first-pass model for
the current stint, up to the current lap. If the model has been assigning P=0.7 for 3
consecutive laps but the driver hasn't pitted, something strategic is suppressing the pit
(track position, safety car window, undercut defence). This residual isn't captured by
TyreLife alone.

**Note**: `TyreLife` already encodes "time since last pit" perfectly (TyreLife = N means
pitted N laps ago). The novel part is the *model-predicted probability residual* — not the
raw tyre age.

**Implementation**: Run a first-pass model to generate OOF pit probabilities, then for
each driver/race sequence compute `cumulative_prob_no_pit = product(1 - P_i)` or
`sum_overdue_prob = sum(P_i for laps in current stint where no pit happened)`.
At test time: predict rows in lap-order within each driver/race, feed running prediction
back as a feature for the next lap. Requires sequential inference, not batch.

**Complexity**: High — needs two model passes and sequential test-time inference. Worth
attempting if we hit a hard ceiling with batch approaches.

### LGBM GPU — not compatible

LightGBM GPU (OpenCL) has a hard bin limit of 256. The `Driver` categorical has 887
unique values → `bin size 525 cannot run on GPU`. No workaround without degrading
model quality. LGBM must stay on CPU. XGBoost and CatBoost use `device="cuda"` /
`task_type="GPU"` which have no such categorical bin restriction.

### ⚠️ Driver features — mixed result by model (2026-05-06)

Ablation (`src/ablation_driver.py`) and full retrain without Driver-identity columns:

| Model | With Driver | No Driver | Delta |
|-------|-------------|-----------|-------|
| LGBM | 0.9474 | **0.9494** | **+0.0020** |
| XGBoost | 0.9489 | 0.9486 | −0.0003 (flat) |
| CatBoost | 0.9479 | 0.9471 | −0.0008 |
| MLP | 0.9457 | 0.9459 | +0.0002 (flat) |
| **Ensemble** | **0.9500** | **0.9500** | flat |

Dropped columns (now excluded via `DRIVER_COLS` constant in `features.py`):
- `Driver` (raw categorical, 887 unique values)
- `driver_pit_rate`, `driver_compound_pit_rate`, `driver_median_tyre_life_at_pit`

**Why mixed**: LGBM was memorising per-driver patterns (overfitting); removing Driver
frees its capacity. CatBoost uses ordered target statistics for categoricals — a
fundamentally different (and more regularised) encoding that was extracting genuine
signal from Driver rather than memorising. XGBoost and MLP are neutral.

**Current state (2026-05-06)**: `DRIVER_COLS = frozenset({"Driver"})` — only the raw
string Driver categorical is excluded. The numeric driver features (`driver_pit_rate`,
`driver_compound_pit_rate`, `driver_median_tyre_life_at_pit`) are kept and contribute
positively to all models.

**Drop Driver string only result**: retrained all 4 models → ensemble **0.9506** (+0.0006
vs previous best). LGBM 52.7% + XGBoost 23.9% + MLP 23.4% + CatBoost 0%.

**Key insight**: dropping all driver features was wrong — the numeric driver statistics
are computable on test and are useful. Only the raw Driver categorical (887 values, 86
train-only drivers) should be excluded.

### Tried & flat (v2, v3, v6, v8)
- v2 polynomial/interaction transforms: −0.0001 delta. LightGBM finds these internally.
- v3 group aggregates (driver/race/compound pit rates + median TyreLife at pit): −0.0002.
  Even with 887 drivers and Laplace smoothing, no gain.
- v6 sequential features (lag1/2/3 LapTime_Delta, roll7 LapTime_Delta, roll5 LapTime (s),
  pace anomaly): −0.0003. Deep trees (num_leaves=490) already capture this internally.

**Plateau diagnosis**: Three versions stall at 0.9431–0.9433. The likely culprit is the
2023 anomaly — 31% of training rows (136k) carry near-zero pit rates. Even with `is_2023`
flagging, the model trains on misleading signal for over a quarter of its data.

**Dropping 2023 result (v4)**: OOF AUC crashed to 0.9147 (−0.029). Confirmed —
2023 must stay in training. The `is_2023` flag lets the model *use* the 2023 rows
constructively: it learns "when is_2023=1, predict near-zero", freeing the rest of the
model to focus on genuine pit patterns. Removing those rows also removes feature-space
coverage and the calibration signal from 136k clean negative examples.

**Revised diagnosis**: The plateau at 0.9431–0.9433 is not caused by 2023 noise.
The model has likely hit the ceiling of what row-level features can provide.
Next levers to try: (1) LightGBM hyperparameter tuning (num_leaves, min_data_in_leaf,
lambda), (2) within-stint rolling features (cumulative lap time relative to stint start),
(3) XGBoost / CatBoost comparison. **v3 remains best submission**.

---

## SHAP + Error Analysis (v5 model, 2026-05-05)

Full analysis in `notebooks/analysis.py`, PNGs in `results/analysis/`.

### SHAP Feature Importance (top features)
1. `is_2023` (0.794) — dominates; model spends most capacity on the 2023 anomaly
2. `Stint` (0.576)
3. `LapTime_Delta` (0.396) — **beats raw TyreLife** — lap-to-lap pace change is key
4. `TyreLife` (0.369)
5. `Year` (0.342) — partly a proxy for 2023
6. `race_progress_x_stint` (0.203), `TyreLife_sq` (0.202), `race_compound_pit_rate` (0.200)

### Calibration
Near-perfect across all bins. No value in post-hoc calibration (isotonic/Platt).

### AUC by Year — CRITICAL: headline AUC is inflated
Full per-model per-year breakdown from `notebooks/distribution_shift.py` (2026-05-06):

| Year | N | pit% | LGBM | XGB | CatBoost | MLP | Ensemble |
|---|---|---|---|---|---|---|---|
| 2022 | 82,989 | 26.7% | 0.9090 | 0.9112 | 0.9109 | 0.9086 | 0.9139 |
| 2023 | 136,147 | 1.0% | 0.9383 | **0.9428** | 0.9335 | **0.9126** | 0.9422 |
| 2024 | 127,110 | 29.5% | 0.9244 | 0.9269 | 0.9255 | 0.9234 | 0.9284 |
| 2025 | 92,894 | 28.4% | 0.9239 | 0.9261 | 0.9248 | 0.9220 | 0.9275 |
| **Overall** | 439,140 | 19.9% | 0.9474 | 0.9489 | 0.9479 | 0.9457 | 0.9500 |

OOF inflation (overall OOF − non-2023 weighted AUC): ~+0.027 across all models.
**Expected LB ≈ 0.929 (ensemble)** given test has same year distribution as train.

⚠️ **MLP on 2023**: AUC = 0.9126 vs trees 0.9335–0.9428. MLP is far weaker at exploiting
the easy 2023 anomaly. This drags down MLP's overall OOF but may not matter on LB if
the ensemble weight given to MLP on 2023 rows is sub-optimal. A conditional ensemble
(higher MLP weight on non-2023 rows) could be worth exploring.

⚠️ **Corrected prior**: "is_2023 irrelevant at test time" was WRONG — test has 30.9%
2023 rows (same as train). The 2023 lever is equally useful at test time.

Year 2022 is the hardest year — first season of major regulation changes, more strategy
variance. Worth investigating separately.

### AUC by TyreLife decile — monotone degradation
| TyreLife | AUC |
|---|---|
| 1–3 laps (fresh) | **0.8918** — hardest; undercuts, safety cars, pure strategy |
| 28–77 laps (old) | **0.9509** — easiest; obvious necessity |

### AUC by Compound
- WET: 0.8620 (n=1355 — too small to learn)
- SOFT: 0.9260, HARD: 0.9288, INTERMEDIATE: 0.9334, MEDIUM: 0.9503

### Hardest Races
Spanish GP (0.9071), Emilia Romagna GP (0.9090), Bahrain GP (0.9132),
Mexico City GP (0.9167, low pit rate 9.1%)

### Implications for next steps
1. **Ensemble (CatBoost/XGBoost)** — most reliable improvement; different handling of
   categoricals (Race, Driver) may improve 2022/HARD/fresh-tyre failure zones
2. **Longer LapTime_Delta rolling windows** — #3 feature; 3-lap window exists, try 5/7
   or exponential weighting — specifically targets fresh-tyre regime
3. **Expected LB ≈ 0.929 (ensemble)** — the realistic ceiling given test mirrors train's
   year distribution. Improvements in Year 2022 and fresh-tyre regimes remain highest-value.
4. **Conditional ensemble (is_2023 split)**: trees dominate on 2023 rows (AUC 0.9335–0.9428
   vs MLP's 0.9126); on non-2023, all models within 0.003. Routing 2023 rows exclusively
   to trees and non-2023 rows to the full ensemble could lift the final blended score.
5. **Novel (Driver, Race, Year) stints in test (1,364 combos)**: trees risk memorising
   driver/race patterns that don't transfer to these unseen combinations. MLP's heavy
   regularisation is an advantage here — likely the main driver of MLP's relative LB
   outperformance vs its OOF ranking.

### Categorical encoding — MLP (2026-05-06)

Tree models handle categoricals correctly:
- LGBM: Pandas `category` dtype → native categorical splits
- XGBoost: `enable_categorical=True` + `category` dtype → partition splits
- CatBoost: `cat_features` + raw strings → ordered target statistics

**MLP bug (now fixed)**: `Compound` and `Race` were label-encoded to integers then
StandardScaled, treating them as continuous ordinal features. This is wrong (e.g. HARD=2
implies "twice as much compound" as SOFT=1).

**Fix applied in mlp_v6**: `pd.get_dummies()` replaces LabelEncoder → 5 Compound + 26
Race binary columns (67 total features vs 38 before). StandardScaler still applied to OHE
columns (centering binary features helps gradient flow).

**Result**: mlp_v6 OOF 0.9455 vs v5 0.9458 — slightly worse because Optuna params were
tuned on the old 38-feature setup. Encoding is now correct but **needs Optuna re-tuning**
to benefit from the extra Race/Compound signal.

**Train/test overlap confirmed**: Race (26), Compound (5), Year (4) are all fully present
in both splits — no unseen categories at inference time. Only `Driver` had train-only values.

### GPU memory fix — MLP (2026-05-06)

`train_fold()` was not freeing GPU tensors between folds, causing CUDA memory fragmentation
that could destabilise the driver under sustained training. Fix: added `del model,
optimizer, scheduler, X_tr_t, y_tr_t, X_val_t, X_test_t` at the end of `train_fold()`
and `torch.cuda.empty_cache()` after each fold in `main()`.

### Ensemble findings (2026-05-05)
- LGBM + CatBoost blend (defaults) gives +0.0005 OOF AUC over LGBM alone; XGBoost defaults zeroed out
- **After GPU tuning**: XGBoost jumped from 0.9456 → 0.9489 and became the dominant model (50.4% weight)
- CatBoost tuning: 0.9471 → 0.9479 (+0.0008); smaller gain — CatBoost less sensitive to HPO here
- **3-model best: 0.9495** (XGBoost 50.4% + CatBoost 31.6% + LGBM 17.9%), +0.0010 vs default blend
- XGBoost GPU tuning was ~10–30s/trial (vs CatBoost ~90–150s/trial) — much faster on RTX 2060
- **PyTorch MLP (GPU)**: standalone OOF 0.9448 — weaker than trees, but contributes 22.9% ensemble weight
  because its errors are decorrelated from all three GBDT models
- **4-model best: 0.9499** (XGBoost 48.1% + MLP 22.9% + CatBoost 19.8% + LGBM 9.2%), +0.0004 vs 3-model
- Simple average of 4 models (0.9498) nearly matches optimized blend — models well-calibrated
- **MLP Optuna tuning (50 trials, 3-fold)**: best params = 4-layer [1024→683→456→304], dropout=0.35, lr=4.8e-3, wd=1.9e-3; 3-fold best 0.9442, 5-fold retrain 0.9457 (+0.0005 vs v2)
- **4-model best (mlp_v3): 0.9500** (XGBoost 46.4% + MLP 26.6% + CatBoost 18.1% + LGBM 8.9%) — crossed 0.95 threshold
- **After dropping Driver (v9/v2/v2/v4): 0.9500** (LGBM 54.8% + MLP 25.4% + XGBoost 19.8% + CatBoost 0%) — OOF flat but LGBM now dominant; CatBoost zero-weighted

---

## Modelling Notes

- **Baseline**: LightGBM 5-fold stratified CV (`src/baseline.py`)
- **CV scores**: logged to `results/cv_scores.csv`
- Categorical columns: `Compound`, `Race` — passed as `category` dtype to LGBM/XGBoost, as `cat_features` to CatBoost, as OHE to MLP (`Driver` excluded via `DRIVER_COLS`)
- `Year` treated as numeric; `is_2023` flag handles the anomaly — this is sufficient

---

## ⚠️ Bug Fixed: ensemble.py test ID ordering

`features.py::build_features()` sorts all rows by `["Driver", "Race", "Year", "LapNumber"]`.
All model scripts call `build_features()` on test data, so `test_{model}.npy` predictions
are in **sorted order**. But `ensemble.py` was loading test IDs directly from the raw CSV
(original order), causing a complete ID/prediction mismatch → 0.5 AUC on Kaggle.

**Fix**: `ensemble.py` now loads test IDs via `build_features(...)["id"]` so the ID order
matches the npy arrays. Individual model submission CSVs were never affected — they extract
IDs from the feature-engineered frame. Re-run `ensemble.py` before any future submission.

---

## Experiments Log

| Date | Script | Description | OOF AUC |
|---|---|---|---|
| 2026-05-04 | baseline_lgbm_v1 | LightGBM 5-fold, raw features + `is_2023` + `TyreLife_sq` | **0.9433** |
| 2026-05-04 | baseline_lgbm_v2 | v1 + `TyreLife_log`, `compound_ord`, `tyre_life_x_compound`, `race_progress_x_stint`, `degradation_rate`, `lap_time_delta_roll3` | **0.9432** (flat) |
| 2026-05-04 | baseline_lgbm_v3 | v2 + 5 group aggregates: `driver_pit_rate`, `driver_compound_pit_rate`, `race_compound_pit_rate`, `driver/race_compound_median_tyre_life_at_pit` (α=20 smoothing, excl. 2023) | **0.9431** (flat) |
| 2026-05-04 | baseline_lgbm_v4 | v3 but DROP 2023 from training entirely | **0.9147** (−0.029 — worse!) |
| 2026-05-05 | baseline_lgbm_v5 | v1 features + Optuna-tuned params (50 trials, 3-fold): num_leaves=490, lr=0.023, min_child_samples=146, reg_lambda=4.5 | **0.9480** (+0.0047 — new best) |
| 2026-05-05 | baseline_lgbm_v6 | v5 + 7 sequential features: lag1/2/3 of LapTime_Delta, roll7 LapTime_Delta, roll5 LapTime (s), pace anomaly (LapTime vs roll5) | **0.9477** (−0.0003 — flat) |
| 2026-05-05 | baseline_lgbm_v7 | same as v5, adds npy saving for ensemble | **0.9477** |
| 2026-05-05 | catboost_v1 | CatBoost GPU, sensible defaults (depth=8, lr=0.05, l2=5) | **0.9471** |
| 2026-05-05 | xgboost_v1 | XGBoost GPU, sensible defaults (max_depth=9, lr=0.05, reg_lambda=5) | **0.9456** |
| 2026-05-05 | ensemble_lgbm_catboost_xgboost_v1 | Optimized blend: LGBM 56.5% + CatBoost 43.5% + XGBoost 0% | **0.9485** (+0.0005 vs LGBM) |
| 2026-05-05 | catboost_v1 (tuned) | GPU Optuna 50-trial tune: depth=9, lr=0.060, l2=2.43, bagging_temp=0.10, rand_strength=0.010, min_leaf=41 | **0.9479** (+0.0008 vs default) |
| 2026-05-05 | xgboost_v1 (tuned) | GPU Optuna 50-trial tune (best at trial 33, AUC 0.9486): now strongest individual model | **0.9489** (+0.0033 vs default) |
| 2026-05-05 | ensemble_lgbm_catboost_xgboost_v2 | Tuned blend: XGBoost 50.4% + CatBoost 31.6% + LGBM 17.9% | **0.9495** (+0.0010 vs v1) |
| 2026-05-05 | baseline_lgbm_v7 (+TyreLife_frac) | Added `TyreLife_frac` = TyreLife / compound median TyreLife at pit | **0.9475** (−0.0002 — flat) |
| 2026-05-05 | mlp_v1 | PyTorch GPU MLP (512→256→128, BN+Dropout, AdamW, cosine LR, early stop) | **0.9448** |
| 2026-05-05 | ensemble_lgbm_catboost_xgboost_mlp_v1 | 4-model blend: XGBoost 48.1% + MLP 22.9% + CatBoost 19.8% + LGBM 9.2% | **0.9499** (+0.0004 — new best) |
| 2026-05-05 | mlp_v2 | mlp_v1 + Yeo-Johnson on numeric cols, StandardScaler on label-encoded cats | **0.9452** (+0.0004 vs v1) |
| 2026-05-05 | ensemble w/ mlp_v2 | XGBoost 47.7% + MLP 24.6% + CatBoost 18.9% + LGBM 8.7% | **0.9499** (flat — MLP gain too small) |
| 2026-05-05 | mlp_v3 | mlp_v2 + Optuna 50-trial tune: 4-layer [1024→683→456→304], dropout=0.35, lr=4.8e-3, wd=1.9e-3 | **0.9457** (+0.0005 vs v2) |
| 2026-05-05 | ensemble w/ mlp_v3 | XGBoost 46.4% + MLP 26.6% + CatBoost 18.1% + LGBM 8.9% | **0.9500** (+0.0001 — new best, crossed 0.95) |
| 2026-05-05 | stacking_v1 | Logistic regression meta-learner (logit inputs, nested 10-fold/5-fold CV, L2 tuned) | **0.9500** (flat — converged to same weights as Nelder-Mead; models well-calibrated) |
| 2026-05-05 | baseline_lgbm_v8 | v7 + 4 sequential features: `window_gap`, `laps_past_window`, `est_laps_remaining`, `pace_anomaly_pct` | **0.9474** (flat, −0.0001 — trees already learn these internally) |
| 2026-05-06 | mlp_v4 | Ablation: drop ALL driver columns (Driver + driver_pit_rate + driver_compound_pit_rate + driver_median_tyre_life_at_pit) | **0.9459** (flat) |
| 2026-05-06 | baseline_lgbm_v10 | Drop ONLY Driver string; keep driver numeric features | **0.9500** |
| 2026-05-06 | xgboost_v3 | Drop ONLY Driver string; keep driver numeric features | **0.9493** |
| 2026-05-06 | catboost_v3 | Drop ONLY Driver string; keep driver numeric features | **0.9476** |
| 2026-05-06 | mlp_v5 | Drop ONLY Driver string (DRIVER_COLS = {"Driver"}); keep driver numeric features | **0.9458** (flat) |
| 2026-05-06 | ensemble w/ mlp_v5 | LGBM 52.7% + XGBoost 23.9% + MLP 23.4% + CatBoost 0% | **0.9506** (+0.0006 — new best) |
| 2026-05-06 | mlp_v6 | mlp_v5 + one-hot encoding for Compound (5) and Race (26); 38→67 features; params not re-tuned | **0.9455** (−0.0003 vs v5 — needs Optuna re-tune) |
| 2026-05-06 | ensemble w/ mlp_v6 | LGBM 53.2% + XGBoost 22.6% + MLP 24.2% + CatBoost 0% | **0.9506** (flat) |
