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

**Rank-blend 0.65·tuned LGBM (`lgbm_pr5_tuned`) + 0.35·RealMLP OOF AUC: 0.959169**
(2026-10-05) ← current best. `w=0.35` was fit (honest half-split) against the GBDT blend in
PR-2 and carried over, not refit. Best single: tuned LGBM 0.959002. No lever has cleared the
+0.0005 bar; this is the sum of two sub-bar but consistent gains (PR-2 + PR-5).
Previous: equal rank-blend LGBM+XGB+CatBoost 0.95894.

| Model | OOF AUC | Script | Notes |
|---|---|---|---|
| LGBM v1 | 0.95874 (CUDA) / 0.95877 (CPU, `lgbm_pr1_base`) | `src/baseline.py` | best_iter only ~170-250 at lr 0.05 |
| XGBoost v1 | 0.95862 | `src/train_xgboost.py` | GPU, ~1 min total; best_iter 550-820 |
| CatBoost v1 | 0.95822 | `src/train_catboost.py` | GPU, fixed 2000 iters, no ES |
| LGBM PR-5 tuned | 0.959002 | `src/tune_lgbm.py --evaluate` | CPU, lr 0.0135, ~700-1050 trees |
| RealMLP v1 | 0.95863 | `src/train_realmlp.py` | GPU, 16 epochs (best at 4-5), ~3.5 min/fold; corr 0.972 with GBDTs; PR-2 sub-bar (+0.00017 honest) |

OOF correlations: LGBM↔XGB 0.9987, LGBM↔Cat 0.9975, XGB↔Cat 0.9981 — the three GBDTs are
near-copies; blending buys only +0.0002. Diversity has to come from a different *kind* of leg.

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
- **GPU**: LightGBM via `get_lgbm_device()` (CUDA probe, CPU fallback). XGBoost `device=cuda`
  (`S6E10_XGB_DEVICE`); CatBoost `task_type=GPU` (`S6E10_CATBOOST_DEVICE`,
  `S6E10_CATBOOST_GPU_RAM_PART`, default 0.5) — runs at ~84% util / 4.5 GB with 0.8.
- **Shared matrices**: every leg builds X / y / X_test via `src/matrices.py::build_matrices()`
  (identical features + category codes; `is_original` column constant 0 unless augmenting).
- **GPU memory rule (all PyTorch scripts)**: `del` model + optimizer + tensors before returning
  from each fold; `torch.cuda.empty_cache()` after each fold/trial.
- **Categoricals**: pandas `category` with a shared train∪test level set for LGBM/XGB;
  `cat_features` for CatBoost; OHE for nets.

---

## Final submission selection

Use the `select-finals` skill: rank by **OOF CV, not public LB**; two diverse finals.

---

## Pre-registrations

### PR-1 — original-data augmentation (registered 2026-10-05, before any number)

- **Arm**: append all 129,880 original rows to each fold's *training* split (never val/test),
  with an `is_original` flag (1 on original, 0 on synthetic and test). Equal sample weight.
- **Control**: the identical script with no original rows (`is_original` constant 0), same
  params, same device (CPU), same folds — a paired read, not a comparison against the
  CUDA-run `lgbm_v1` number.
- **Instrument**: `src/train_lgbm_orig.py`, LGBM v1 params, 5-fold stratified seed 42, one
  model seed. Paired per-fold AUC delta (aug − base).
- **Bar (PASS)**: OOF delta ≥ **+0.0005** AND positive in **≥ 4/5 folds**. Anything else is a
  FAIL for this arm (equal-weight, flag-only); a FAIL does not close weighted / delay-dropped
  variants.
- **Prior**: ~40% PASS. For: classic Playground lever, original is near-separable so its
  labels are cleaner. Against: delays drifted ~12× (original rows are off-distribution) and
  700k synthetic rows already dwarf 130k originals.
- **RESULT (2026-10-05): FAIL — a clean null.** base 0.958768 / aug 0.958764, OOF delta
  **−0.000004**; per-fold deltas [+8, −7, −5, −5, +6]e-5 (2/5 positive). OOF correlation
  base↔aug 0.9993. The control reproduced the CUDA `lgbm_v1` (0.95877 vs 0.95874), so the
  instrument is sound. Reading: the original's clean labels don't transfer — whatever the
  generator did to the label/feature relationship, 130k rows from the source distribution add no
  information the 700k synthetic rows lack. **Closed: equal-weight, flag-only augmentation.**
  **Still open** (cheaper levers first): original as a *feature source* (e.g. per-combo target
  rate looked up from the original — no row mixing, so no surface mixing); down-weighted original
  rows; original with delays dropped.

### PR-2 — RealMLP-TD leg (registered 2026-10-05, before any number)

- **Arm**: `src/train_realmlp.py` — `RealMLP_TD_Classifier` (package defaults + s6e8 overrides:
  `epoch_select="auc"`, `loss_prior_power=0`, `train_bs=512`, AMP), **16 epochs**, seed 42,
  same folds as the GBDTs, v1 features via `build_matrices()`.
- **Control**: the GBDT blend = equal rank-average of `lgbm_pr1_base`, `xgb_v1`, `catboost_v1`
  (0.95894).
- **Instrument**: blend = `(1−w)·gbdt + w·realmlp` on ranks, `w` on a 0.05 grid. **Honest
  fit**: `w` fit on a random half of train, scored on the other half, both directions; the
  delta is the mean of the two held-out deltas (lesson 3). `src/blend_check.py`.
- **Bar (ADOPT)**: honest blend delta ≥ **+0.0005**. Stop-loss (routing only): solo OOF
  < 0.950 → don't spend seeds on it.
- **Caveat**: RealMLP picks its epoch on the val fold it is scored on, so its solo OOF is
  slightly optimistic. The blend delta inherits a small part of that; the bar is set high
  enough to absorb it.
- **Prior**: ~25% ADOPT. Against: s6e8's NN legs earned weight 0.0. For: here the GBDTs are
  0.998-correlated and early-stop at ~200 trees (low-signal, noisy target), a regime where a
  smooth model can genuinely rank differently.
- **RESULT (2026-10-05): FAIL — real but sub-bar.** Solo OOF **0.95863** (folds 0.95888 /
  0.95768 / 0.95918 / 0.95853 / 0.95894 — level with XGB, ~3.5 min/fold on GPU; best epoch 4-5
  of 16 every fold). Corr with the GBDT blend 0.9724 (vs 0.998 among the GBDTs). Honest blend:
  `w = 0.35` in *both* halves, held-out deltas +0.000175 / +0.000157 → **+0.000166** vs the
  +0.0005 bar. The gain is consistent, not noise — but a third of the bar. Not adopted as a
  pre-registered member; the arrays are kept, and it is the natural first member of any later
  stack. Reading: the most decorrelated leg available buys +0.00017, so diversity among
  learners on v1 features is nearly exhausted; the next lever has to change the *information*
  (features / original-as-lookup / target encoding), not the learner.

### PR-3/4/5 — information levers, run in sequence (registered 2026-10-05, before any number)

Common instrument: `src/run_lgbm_arm.py` (PR-3, PR-4) / `src/tune_lgbm.py` (PR-5) — the PR-1
control's LGBM config (CPU, 8 threads, seed 42, same folds), paired per fold against
`oof_lgbm_pr1_base.npy` (0.958768). An `--arm base` run must reproduce it first (instrument
check). **Common bar (PASS)**: OOF delta ≥ **+0.0005** AND ≥ **4/5** folds positive. A PASS
on the LGBM leg is then re-priced at the blend (lesson 7) before adoption.

- **PR-3 — original as a feature source.** `orig_pred` (LGBM fit on the full original,
  predicting synthetic rows) + `orig_rate_<col>` (original satisfaction rate per value of each
  of the 4 categoricals and 13 ratings). Uses no synthetic labels → global, not fold-fitted
  (rule 2). Prior **~20%**: PR-1 showed original rows carry nothing the synthetic rows lack, and
  a model of the original is the same information compressed — but a compressed, near-separable
  view could still be an easier split for the trees.
- **PR-4a — fold-safe TE on the 78 rating pairs** (smoothed (r_i, r_j) target rate, nested
  inner CV, fit per fold on train only; raw pair keys not fed). Prior **~15%**: 700k rows let
  the trees learn pairwise interactions directly; TE mostly helps when combos are sparse.
- **PR-4b — ratings as `category` dtype** (0 = N/A is not ordinal). Prior **~15%**: trees
  already isolate 0 with one split.
- **PR-5 — LGBM tuning.** Optuna TPE, 40 trials, objective = 3-fold AUC on a fixed 50% subsample
  (seed 0) for speed; space lr [0.01, 0.1] log, num_leaves [15, 255], min_child_samples
  [20, 1000] log, reg_lambda [1e-3, 30] log, colsample_bytree [0.3, 1], subsample [0.5, 1],
  n_estimators 5000 + ES 100. Final: 5-fold with the best params, paired as above. Caveat: the
  tuning rows overlap the evaluation rows, so the read carries a small selection optimism
  (40 trials → ~1e-4 at most); the bar absorbs it. Prior **~40%**: the control early-stops at
  ~200 trees with 127 leaves — the shape of an over-aggressive config on a noisy target.
- **Instrument check (2026-10-05): PASS** — `--arm base` reproduced `lgbm_pr1_base` exactly
  (per-fold delta 0.0, corr 1.0000), so CPU LGBM is deterministic here and the arms are paired.
- **PR-3 RESULT: FAIL (null).** OOF 0.958809, delta **+0.000041**, per-fold
  [−3, −2, −18, +36, +8]e-5 (2/5). Corr 0.9979. Together with PR-1: the original carries nothing
  the synthetic rows lack, *in either form* (rows or compressed features). **Closed: the
  original dataset as an information source** (weighted rows remain technically untested but
  have no mechanism left).
- **PR-4a RESULT: FAIL (harmful).** OOF 0.958427, delta **−0.000341**, 0/5 positive (every fold
  −1.2e-4…−4.5e-4). 78 TE columns dilute the column sampling and add encoder noise; the trees
  already resolve rating pairs from 560k rows/fold. **Closed: pairwise rating TE.**
- **PR-4b RESULT: FAIL (null).** OOF 0.958719, delta **−0.000049**, 1/5. **Closed: ratings as
  categoricals.**
- **PR-5 RESULT: FAIL on magnitude — consistent but sub-bar.** Search: best 0.95774 vs control
  trial 0.95746 on the 50% subsample (40 trials, ~2 h on CUDA LGBM — CPU would have been faster
  on this narrow data). Best params → `results/best_params_lgbm_pr5.json`: lr 0.0135,
  num_leaves 144, min_child_samples 176, reg_lambda 2.07, colsample 0.69, subsample 0.87.
  5-fold paired: OOF **0.959002**, delta **+0.000234**, per-fold [+33, +13, +15, +33, +25]e-5
  (**5/5**), best_iter 716-1046. Real (every fold) but half the bar.
  **Blend pricing (lesson 7, bookkeeping):** swapping it into the 3-GBDT rank blend gives
  0.958929 (−0.000007 — fully absorbed), but **tuned LGBM + 0.35·RealMLP = 0.959169** (w carried
  from PR-2, not refit) — the best OOF so far, +0.00023 over the 0.95894 blend.

### PR-6 — duplicate / frequency structure (2026-10-05) — CLOSED BY SCREEN, no full arm

Common bar (+0.0005) carried over. Cheap screens against `oof_lgbm_pr5_tuned` (0.959002), run
first because each one *bounds* the effect (gate on magnitude, not on "the structure exists"):
- **Exact near-copies: none.** 0 duplicated full feature vectors in train∪test (999,479 unique
  of 999,479). The "generator emits copies" mechanism does not exist here.
- **Partial twins exist but carry no label information.** 53% of rows share their exact
  ratings+categoricals vector (67% ratings-only). Leave-one-out *residual* of a row's twins
  (y − oof, shrunk by group size) added to the OOF logit: best delta **≤ 0.000000** for every key
  (ratings / +cats / +age) and shrinkage (k = 5, 20), even with the mixing weight picked
  in-sample. The model already knows everything a twin's label could say.
- **Pattern counts (unsupervised density) ≈ calibration noise.** OOF residual by count bin moves
  ±0.001-0.002 in probability (≤ 2 se). In-sample upper bound from re-centring every count bin:
  **+0.000028** (ratings) / **+0.000035** (ratings+cats); count × OOF-decile cells: **+0.000082** /
  **+0.000047** — all hundreds of in-sample fitted cells, so optimistic, and still ≤ 1/6 of the bar.
- **Closed: duplicate/twin labels and pattern-frequency features.**

### PR-7 — RealMLP tuning (registered 2026-10-05, before any number)

- **Stage 1 (screen, fold 1 only)**: configs vs `realmlp_v1` fold 1 (0.95888) — epochs 6 / 8 /
  10; at 8 epochs: width 256 / 1024, lr 0.005 / 0.02, stronger regularisation (dropout 0.15,
  wd 0.05). **Noise control**: the v1 config at seed 43 on fold 1 — a config must beat v1 by
  more than |seed-43 − seed-42| to be called a winner. ≤ 2 winners go to stage 2.
- **Stage 2**: full 5-fold of the winner(s).
- **Bar (ADOPT, at the deliverable — lesson 7)**: `blend(lgbm_pr5_tuned, new) −
  blend(lgbm_pr5_tuned, realmlp_v1)`, each with `w` fit by the honest half-split, ≥ **+0.0005**.
  Solo delta reported but not the decision.
- **Prior ~15%**: tuning a leg raises its solo score, but s6e7/s6e8 both showed a combiner
  absorbs improved legs (lesson 7). For: the schedule is visibly mis-set (peak at epoch 4-5 of
  a 16-epoch cosine — the LR had not decayed when it peaked).
- **RESULT (2026-10-05): FAIL at stage 1 — no config clears seed noise; stage 2 not run.**
  Fold-1 AUC (v1 seed 42 = 0.95888): **seed-43 control 0.95875** (noise band ±0.00013);
  e6 0.95874, e8 0.95885, e10 0.95886, e8·w256 0.95885, e8·w1024 0.95879, e8·lr0.005 0.95877,
  e8·lr0.02 0.95890, e8·reg (dropout 0.15, wd 0.05) 0.95887. Spread of all 9 = 0.00016 ≈ the
  seed gap. **The epoch-5 peak is not the schedule:** every config peaked at epoch 5 whatever
  its length (6, 8, 10, 16) — 560k rows/fold overfit after ~5 passes. Only e8·reg peaked at its
  *last* epoch (its boundary binds), but gate on magnitude: all of v1 bought +0.00017 at the
  blend, so even a solo gain several times the observed spread cannot reach +0.0005 there.
  **Closed: RealMLP hyperparameter tuning** (schedule, width, lr, regularisation).

### Submission log

| Date | File | Recipe | OOF | Note |
|---|---|---|---|---|
| 2026-10-05 | `blend_lgbmt65_realmlp35.csv` | 0.65·rank(`test_lgbm_pr5_tuned`) + 0.35·rank(`test_realmlp_v1`) | 0.959169 | first submission; LB not checked (LB-blind) |

### PR-8 — TabM leg (registered 2026-10-05, before any number)

- **Arm**: `src/train_tabm.py` — pytabkit `TabM_D_Classifier` (defaults; `compile_model`,
  `allow_amp`, batch 2048, lr 0.004 per s6e7), `val_metric_name="1-auc_ovr"` on pytabkit's own
  internal split of the training fold (our val fold untouched), v1 features, seed 42.
- **Control**: the submitted blend `0.65·rank(lgbm_pr5_tuned) + 0.35·rank(realmlp_v1)` (0.959169).
- **Instrument**: `(1−w)·control + w·rank(tabm)`, `w` by the honest half-split
  (`blend_check.py tabm_v1 --control-blend`).
- **Bar (ADOPT)**: honest delta ≥ **+0.0005**. Timebox: fold 1 > 25 min → kill, record as
  priced out.
- **Prior ~10%**: RealMLP already supplies the "NN kind" and bought only +0.00017; TabM is the
  same kind (an MLP ensemble), so it is likely correlated with RealMLP. For: its implicit
  k-member ensemble and different numeric embeddings are the strongest remaining NN variant.

## Next steps (remove each item when implemented)

Closed so far: original as rows (PR-1) and as features (PR-3); pairwise rating TE (PR-4a);
categorical ratings (PR-4b); duplicate/frequency structure (PR-6, by screen); RealMLP tuning (PR-7). Sub-bar but real: RealMLP (PR-2), LGBM tuning (PR-5).
**Untested lever classes:**
1. Tune XGB / CatBoost and an LR stack over all legs (lesson 7 says absorption is likely —
   price at the blend).
2. Different NN family (TabM) for a second in-kind leg.

---

## Experiments log

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-10-05 | scaffold | Seeded dir; pipeline smoke-tested on the original data (not the competition data) | (0.99491, original only) |
| 2026-10-05 | lgbm_v1 | `baseline.py` on the competition data, v1 features, untuned defaults | 0.95874 |
| 2026-10-05 | lgbm_pr1_base / _aug | PR-1 original augmentation, paired on CPU — **FAIL**, delta −0.000004 | 0.95877 / 0.95876 |
| 2026-10-05 | xgb_v1 | XGBoost GPU, s6e8 hand-set params | 0.95862 |
| 2026-10-05 | catboost_v1 | CatBoost GPU, 2000 iters, border 254, max_ctr 1 | 0.95822 |
| 2026-10-05 | blend (OOF only) | equal rank-blend of the three | 0.95894 |
| 2026-10-05 | lgbm_arm_orig | PR-3 original-derived features — **FAIL** (+0.000041, 2/5) | 0.958809 |
| 2026-10-05 | lgbm_arm_te_pairs | PR-4a fold-safe TE on 78 rating pairs — **FAIL**, harmful (−0.000341, 0/5) | 0.958427 |
| 2026-10-05 | lgbm_arm_cat_ratings | PR-4b ratings as category — **FAIL** (−0.000049, 1/5) | 0.958719 |
| 2026-10-05 | lgbm_pr5_tuned | PR-5 Optuna 40 trials — **FAIL** on magnitude (+0.000234, 5/5) | 0.959002 |
| 2026-10-05 | blend (OOF only) | 0.65·lgbm_pr5_tuned + 0.35·realmlp_v1 (rank) | **0.959169** |
| 2026-10-05 | PR-7 sweep | RealMLP fold-1 sweep, 8 configs + seed control — **FAIL**, all within seed noise (±0.00013) | (fold 1 only) |
| 2026-10-05 | realmlp_v1 | PR-2 RealMLP-TD, 16 ep — **FAIL** vs bar, honest blend +0.000166 at w=0.35 | 0.95863 |
