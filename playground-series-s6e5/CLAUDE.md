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

### Tried & flat (v2 + v3)
- v2 polynomial/interaction transforms: −0.0001 delta. LightGBM finds these internally.
- v3 group aggregates (driver/race/compound pit rates + median TyreLife at pit): −0.0002.
  Even with 887 drivers and Laplace smoothing, no gain.

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
