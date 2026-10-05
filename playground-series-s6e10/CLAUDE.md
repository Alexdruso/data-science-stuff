# Predicting Airline Satisfaction (playground-series-s6e10)

**Task**: binary classification — predict `satisfaction` (passenger satisfied vs neutral/dissatisfied)
**Metric**: ROC AUC on the predicted probability. Submission: `id,satisfaction` (probability).
**Deadline**: 2026-10-31 23:59 UTC — **queried from the source** (`kaggle competitions list -s
playground-series-s6e10` and the competition's Timeline page) on 2026-10-05. Start date
2026-10-01. Never carry a deadline that no document sourced: an unsourced "~Aug 2" ran through
22 days of s6e7 planning and the real date was Jul 31.

---

## ⚠️ CRITICAL: Row-order invariant — always load labels and IDs through build_features()

`features.py::build_features()` sorts every dataframe by `["id"]`. All
`results/oof_<model>.npy` and `results/test_<model>.npy` arrays are stored in this sorted order.

`build_features()` also **renames the raw headers to snake_case** and **coerces the target to
Int8 0/1**, so reading `TARGET` from a raw CSV is wrong twice over (order *and* encoding).

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — raw CSV order ≠ npy order, and the raw target may not be 0/1
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

---

## ⚠️ No data leakage — fold-aware rules

1. **Target encoding is fold-aware, NEVER global.** Fit any target-derived feature on the
   fold's training split only (`data_science_stuff.kaggle.encoding`).
2. **Group stats over feature columns** (no per-row targets) may use the full raw train CSV as
   global priors — if used, document that choice here explicitly.
3. **Original-data augmentation** (`data/original/data.csv`): rows may be appended to each
   fold's *training* split only — never to the validation split, or OOF stops measuring the
   synthetic distribution the LB is scored on.

---

## Dataset

Synthetic, generated from the [Airline satisfaction dataset](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k)
("feature distributions close to, but not exactly the same as, the original").

| Split | Rows | Columns |
|---|---|---|
| train | 699,635 | 23 (id + 21 features + target) |
| test  | 299,844 | 22 |
| original | 129,880 | 22 (no id) — `data/original/data.csv` |

Download: `kaggle competitions download -c playground-series-s6e10 -p data/ && cd data &&
unzip -q playground-series-s6e10.zip && rm playground-series-s6e10.zip`. Synthetic headers are
identical to the original's (different column order); target ships as `True`/`False` —
**train positive rate 44.36%** (original 43.45%). Nulls: only `arrival_delay_in_minutes`
(292 train / 130 test). `baggage_handling` has no 0 in train (range 1-5).

### Raw features (snake_case after `build_features`)

- Categorical: `gender` (M/F), `customer_type` (Loyal / disloyal), `type_of_travel`
  (Business / Personal), `class` (Business / Eco / Eco Plus)
- Numeric: `age`, `flight_distance`, `departure_delay_in_minutes`, `arrival_delay_in_minutes`
  (393 NaN in the original)
- 13 service ratings, 0-5: `inflight_wifi_service`, `departure_arrival_time_convenient`,
  `ease_of_online_booking`, `gate_location`, `food_and_drink`, `online_boarding`,
  `seat_comfort`, `inflight_entertainment`, `on_board_service`, `leg_room_service`,
  `baggage_handling`, `checkin_service`, `cleanliness`

Original target: bool, 43.4% positive (56,428 / 129,880).

---

## Key EDA findings

- **Original:** a 0 rating means "not applicable", not "worst" — mean 0.15 zero-ratings per row.
  Kept raw (trees split on it) and counted as `n_zero_ratings`.
- **Original is near-separable:** the v1 baseline scores **OOF AUC 0.99491** on a 100k/30k split
  of the *original* data (pipeline smoke test, 2026-10-05) — not a guide to the synthetic
  set (see next bullet).
- **Synthetic ≠ original — much harder.** Same baseline: 0.99491 on original vs **0.95874** on
  synthetic. The generator destroyed most of the near-deterministic structure (~0.04 AUC of
  label noise / smoothing).
- **Delays collapsed ~12×:** mean departure delay 1.18 min (train) vs 14.7 (original);
  `dep == 0` on 90.8% of train rows vs 56.5%; `arr == dep` on 86.9% vs 47.2%. Delay is a weak
  signal here (sat rate 0.448 at dep=0 vs 0.400 at dep>0). Flight distance also drifted
  (mean 1354 vs 1190); ratings are shifted up ~0.1 across the board. ⇒ original rows are
  off-distribution on delays; augmentation must carry an `is_original` flag and be screened,
  not assumed.
- **No train/test shift:** adversarial validation AUC **0.4994** (raw columns, 5-fold LGBM).
  Category marginals match to 3 decimals. OOF should rank candidates well, down to its noise floor.
- Only 17 train rows exactly duplicate an original row — no lookup leak.
- Zero ratings ("N/A"): 6.4% of train rows have ≥1 vs 7.9% in original.
- TODO: missingness *mechanism* of `arrival_delay_in_minutes` (`kaggle.shift.mask_mechanism_report`
  — s6e7 lesson 8) — only 292 rows, so low priority.

---

## Current best

**LGBM v1 OOF AUC: 0.95874** (2026-10-05) ← current best (untuned; folds 0.95778-0.95940)

| Model | OOF AUC | Script |
|---|---|---|
| LGBM v1 | 0.95874 | `src/baseline.py` |

---

## Feature engineering

`src/features.py::build_features()` (v1): raw columns + `rating_mean/min/max`,
`n_zero_ratings`, `n_high_ratings` (≥4), `delay_recovered` (arr − dep), `delay_max`,
`delay_per_mile`.

---

## Modelling notes

- **Shared utilities live in `data_science_stuff.kaggle`** — import, never copy.
- **CV**: `run_cv` (5-fold stratified, random_state=42). Scores logged to `results/cv_scores.csv`.
- **OOF/test arrays store the positive-class probability** (1-D). AUC is rank-based: no
  threshold/decision layer.
- **GPU**: LightGBM via `get_lgbm_device()` (CUDA probe, CPU fallback).
- **GPU memory rule (all PyTorch scripts)**: `del` model + optimizer + tensors before returning
  from each fold; `torch.cuda.empty_cache()` after each fold/trial.
- **Categoricals**: pandas `category` with a shared train∪test level set for LGBM/XGB;
  `cat_features` for CatBoost; OHE for nets.

---

## Final submission selection

Use the `select-finals` skill: rank by **OOF CV, not public LB**; two diverse finals.

---

## Next steps (remove each item when implemented)

1. Original-data augmentation (train-split only, with an `is_original` flag) — the classic
   Playground lever; pre-register the bar before running.
2. XGBoost / CatBoost legs (`add-model`), then tune the strongest leg — s6e8 showed one tuned
   leg can beat a whole untuned blend.
3. Treat ratings as categoricals (CatBoost native / TE on rating crosses) — the 0 = N/A
   semantics make them non-ordinal at one end.

---

## Experiments log

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-10-05 | scaffold | Seeded dir; pipeline smoke-tested on the original data (not the competition data) | (0.99491, original only) |
| 2026-10-05 | lgbm_v1 | `baseline.py` on the competition data, v1 features, untuned defaults | 0.95874 |
