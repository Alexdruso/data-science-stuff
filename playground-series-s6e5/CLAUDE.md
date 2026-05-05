# PS S6E5 — F1 Pit Stop Prediction

**Task**: Binary classification — predict `PitNextLap` (will this driver pit on the next lap?)
**Metric**: AUC-ROC
**Deadline**: May 31, 2026

> **Keep this file up to date.** Append new findings from experiments, feature
> engineering, or model runs under the relevant section. This is the single
> source of truth for what we know about this competition.

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

### Train / Test Distribution

No meaningful distributional shift across any feature. Year and Compound proportions
match to within <1 percentage point.

---

## Feature Engineering Ideas

- `TyreLife × Compound` interaction (SOFT degrades faster per lap)
- `TyreLife² ` or log-TyreLife (non-linear degradation curve)
- `RaceProgress × Stint` (late-race stint flags strategic windows)
- Binary `is_2023` flag to isolate the anomaly year
- `LapTime_Delta` rolling mean (smooth out noise)
- `Cumulative_Degradation / TyreLife` ratio (degradation rate)

### Tried & flat (v2, v3, v6)
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
| Year | AUC | Pit rate | N |
|---|---|---|---|
| 2022 | **0.9099** | 26.7% | 82,989 |
| 2023 | 0.9391 | 1.0% | 136,147 |
| 2024 | 0.9251 | 29.5% | 127,110 |
| 2025 | 0.9251 | 28.4% | 92,894 |

OOF AUC of 0.9480 is inflated by the easy 2023 prediction. **Effective real-world AUC
(non-2023 rows weighted) ≈ 0.922**. The model's biggest lever (`is_2023`) is irrelevant
at test time if test rows are all non-2023.

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
3. **The real competition ceiling is ~0.922**, not 0.948 — improvements in Year 2022
   and fresh tyres are the highest-value targets

### Ensemble findings (2026-05-05)
- LGBM + CatBoost blend gives +0.0005–0.0007 OOF AUC over LGBM alone
- XGBoost with default params adds no value (zeroed out by optimizer every time)
- **Next step: GPU-tune CatBoost + XGBoost** (~1–2 min/trial on RTX 2060 vs 8–10 min on CPU)
  — full 50-trial Optuna run now feasible in ~1 hour each

---

## Modelling Notes

- **Baseline**: LightGBM 5-fold stratified CV (`src/baseline.py`)
- **CV scores**: logged to `results/cv_scores.csv`
- Categorical columns: `Driver`, `Compound`, `Race` — passed as `category` dtype to LightGBM
- `Year` is currently treated as numeric — consider ordinal or one-hot given the 2023 anomaly

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
