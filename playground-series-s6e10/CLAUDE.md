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

**2026-10-09: `hybrid_lrstack_1009.csv`, OOF 0.962162 (submitted)** — LR stack of 199 public OOF
pairs + 6 own k10 legs (RealMLPs now honest). Hedge `hybrid_lrstack_1009_nomlp.csv` 0.962107
(no public MLP members, which are probably epoch-selection-inflated) also submitted.

**2026-10-08: `blend_own_1008b.csv`, honest OOF 0.961695 (submitted)** — adds the original-data
teacher logit + route profile (PR-19) and aux-task expected ratings (PR-20) to the tree legs,
aux features to RealMLP. Own work only. A hybrid with the public OOF library
(`hybrid_lrstack_v1.csv`, 0.962101) is written but not submitted (PR-21, user's call).

**2026-10-07 (end of day): `blend_final_1007_orig2.csv`, honest OOF 0.961435** — route
features (PR-12) + residual route TE (PR-13) + RealMLP fd/age embeddings (PR-15) + original
rows appended to training folds (PR-16) in both legs. See the submission log.

Earlier on 10-07: rank-blend 0.475·`lgbm_pr12_route` + 0.525·`realmlp_route` — OOF 0.960932
(honest +0.00176 over the row below; PR-12). Legs: RealMLP+route 0.96065, tuned LGBM+route
0.960522, XGB+route 0.96020. The table below predates PR-12.

Previous:

**Rank-blend 0.65·tuned LGBM (`lgbm_pr5_tuned`) + 0.35·RealMLP OOF AUC: 0.959169**
(2026-10-05) ← current best. `w=0.35` was fit (honest half-split) against the GBDT blend in
PR-2 and carried over, not refit. Best single: tuned LGBM 0.959002. No lever has cleared the
+0.0005 bar; this is the sum of two sub-bar but consistent gains (PR-2 + PR-5).
Previous: equal rank-blend LGBM+XGB+CatBoost 0.95894.

| Model | OOF AUC | Script | Notes |
|---|---|---|---|
| TabM v1 | 0.95815 | `src/train_tabm.py` | GPU, ~18 min/fold; PR-8 null (+0.000006 at the blend) |
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
| 2026-10-09 | `hybrid_lrstack_1009.csv` | LR (C 0.3) on 199 public pairs + 6 own k10 legs (honest RealMLP) | 0.962162 | first submission using public predictions (user: objective winning); LB not checked |
| 2026-10-09 | `hybrid_lrstack_1009b_nomlp.csv` | hedge v2: pool 214 (+15 members from public notebook outputs, `data/public_oof/kout_norm`) without public MLPs + 6 own legs, C 0.1 | 0.962135 | best robust candidate; full-pool version `hybrid_lrstack_1009b.csv` 0.962161 = unchanged, not submitted |
| 2026-10-09 | `hybrid_lrstack_1009_nomlp.csv` | same, without the 53 public MLP members (inflation hedge) | 0.962107 | robust candidate; LB not checked |
| 2026-10-08 | `blend_own_1008b.csv` | honest-NM 10 legs: + teacher/route-profile LGBM, aux-task LGBMs (incl. extra-trees), RealMLP + aux / + og,agg (`blend_nm.py`) | 0.961717 | honest **0.961695** (+0.00026 vs 10-07); own work only, no public predictions; LB not checked |
| 2026-10-05 | `blend_lgbmt65_realmlp35.csv` | 0.65·rank(`test_lgbm_pr5_tuned`) + 0.35·rank(`test_realmlp_v1`) | 0.959169 | first submission; LB not checked (LB-blind) |
| 2026-10-07 | `blend_final_1007_orig2.csv` | honest-NM: `realmlp_route_ls0_emb4_fdcat_agecat_orig` 0.52 + `lgbm_pr13_resid_orig` 0.17 + `lgbm_pr12_route_orig` 0.17 + ls0_emb4_fdclass 0.10 + ls0 0.05 | 0.961441 | honest **0.961435** — current best; LB not checked |
| 2026-10-07 | `blend_final_1007_orig.csv` | 9-leg honest-NM: `realmlp_route_ls0_emb4_fdcat_agecat_orig` 0.54 + `lgbm_pr13_resid` 0.26 + 7 RealMLP variants (≤0.07 each) | 0.961401 | honest **0.961395** (+0.000118 vs `blend_final_1007`); LB not checked |
| 2026-10-07 | `blend_final_1007.csv` | 8-leg rank blend, honest-NM weights (avg of halves): `lgbm_pr13_resid` 0.30 + RealMLP route variants (ls0_emb4 0.16, delaycat 0.12, ls0_emb4_fdclass 0.11, ls0 0.10, ls0_emb2 0.09, fdclass 0.07, emb24 0.05) | 0.961284 | honest 0.961277 (+0.000034 vs submission 3 — NN-variant breadth); LB not checked |
| 2026-10-07 | `blend_lgbmresid35_mlpage30_mlpdelay35.csv` | 0.35·rank(`lgbm_pr13_resid`) + 0.30·rank(`realmlp_route_fdcat_agecat`) + 0.35·rank(`realmlp_route_fdcat_agecat_delaycat`) | 0.961247 | honest 4-leg fit 0.961243 (+0.00031 vs submission 2); LB not checked |
| 2026-10-07 | `blend_route_lgbm_realmlp_w52.csv` | 0.475·rank(`test_lgbm_pr12_route`) + 0.525·rank(`test_realmlp_route`) | 0.960932 | PR-12 route features in both legs; honest delta +0.001761 vs the 10-05 recipe (w 0.55/0.50 per half) |

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
- **RESULT (2026-10-05): FAIL (null).** Solo OOF **0.95815** (folds 0.95829 / 0.95746 / 0.95878 /
  0.95807 / 0.95837; ~17-19 min/fold, early-stops ~epoch 40) — the weakest leg. Corr with the
  submitted blend 0.9718. Honest blend: w = 0.05 in both halves → **+0.000006**. RealMLP already
  holds the NN channel. **Closed: a second MLP-family leg.**

### PR-9 — LR stack over all legs (registered 2026-10-05, before any number)

- **Arm**: `src/stack_lr.py` — logistic regression on the logits of every finished leg
  (`lgbm_pr5_tuned`, `lgbm_pr1_base`, `xgb_v1`, `catboost_v1`, `realmlp_v1`, + `tabm_v1` if it
  has finished), standardised, fit with an **outer 5-fold CV over the OOF matrix** (different
  seed from the base folds) so the stacker's own OOF is honest. C on a small grid chosen inside
  each outer fold.
- **Control**: the submitted blend (0.959169), scored on the same rows.
- **Bar (ADOPT)**: stack OOF − control ≥ **+0.0005**. Prior **~10%**: lesson 7 — the combiner
  absorbs; with 0.997+ correlated legs an LR has almost nothing a rank-average lacks.
- **RESULT (2026-10-05, 5 legs, before TabM): FAIL (null).** Stack OOF **0.959174** vs control
  0.959169 → **+0.000004**. Final C = 0.01 (strongest shrinkage on the grid); standardised
  coefs lgbm_pr5 0.54, lgbm_v1 0.45, xgb 0.27, cat 0.33, realmlp 1.36 — the LR rediscovers the
  rank blend. Re-run with TabM once it lands (pre-registered).
- **Re-run with TabM (6 legs): FAIL (null).** Stack OOF **0.959172** vs 0.959169 → **+0.000002**
  (C = 0.1). **Closed: LR stacking over these legs.**

### PR-10 — per-row rating-profile features (registered 2026-10-07, before any number)

- **Arms** (`src/run_lgbm_arm.py`, PR-3 instrument, base LGBM config, paired vs
  `lgbm_pr1_base`; each arm ~45 s so all are run):
  * `profile` — rating_std, rating_median, within-row z-score of each of the 13 ratings
    ((r − row mean)/row std), and the 6-bin histogram (count of 0s…5s per row);
  * `segment` — the rating deviations from the row's (type_of_travel × class) segment mean
    of that rating (global, no labels; rule 2), plus a 6-level segment key;
  * `numcat` — Deotte-style: `age`, `flight_distance` and the two delays duplicated as
    `category` + per-value count (frequency) of each; synthetic generators often leave
    value-level artefacts in numerics that a split on the raw value cannot isolate.
- **Bar (PASS)**: OOF delta ≥ **+0.0005** AND ≥ 4/5 folds (unchanged). Screening rule: any
  arm with delta ≥ +0.0002 is re-run on the tuned config (`lgbm_pr5_tuned`) and priced at
  the submitted blend before anything is adopted (lesson 7).
- **Prior**: profile ~10% (trees on 700k rows already see the ratings jointly; PR-4a showed
  added rating columns dilute), segment ~5%, numcat ~15% (the only one aimed at a
  generator artefact rather than at structure the trees can already learn).

### PR-11 — seed bagging of the submitted legs (registered 2026-10-07, before any number)

- **Arm**: RealMLP v1 at seeds 43/44/45 (+42 existing) → `realmlp_v1_s4x`; tuned LGBM at
  seeds 43-45 if the CPU frees up. Blend = submitted recipe with each leg replaced by the
  mean rank of its M seeds.
- **Instrument**: M-slope (lesson 1) — paired delta of blend(M) − blend(1) at M = 1..4,
  fit `|delta| ~ M**b` on all M (not just three points; lesson 4). Report b.
- **Bar (ADOPT)**: delta at M = 4 ≥ **+0.0005**; bookkeeping adoption (it costs no
  information) is allowed at any positive delta since a bagged leg can only be submitted,
  never hurt in expectation — but it is logged as sub-bar.
- **Prior** ~5% to clear the bar: b ≈ −1 (variance reduction) is the expected mechanism;
  RealMLP per-seed fold-1 spread is ±0.00013, so averaging four seeds should buy ~+0.0001.

### PR-12 — flight_distance as a ROUTE key (registered 2026-10-07, after the PR-10 screens)

PR-10 screens (base config, paired vs `lgbm_pr1_base`) found that `flight_distance`
(3,521 distinct values, peaky counts: 594 mi ×11.5k) behaves like a route ID:
`te_fd` (fold-safe TE of fd, fd×class, fd×travel, fd×customer) **+0.001269** 5/5;
`orig_fd` (original's smoothed satisfaction rate + count per fd — no synthetic labels)
**+0.001100** 5/5; `nc_cnt` (value counts) +0.000576 5/5. Age / delays as categories
*hurt* (−0.00047 / −0.00024): the effect is specific to the route key.
- **Arms** (base config first, then tuned config): `route` = te_fd + orig_fd + cnt_fd;
  `route_plus` = route + richer keys (fd×class×travel, fd×customer×travel TE; original
  per-(fd, class) rate).
- **Bar (ADOPT)**: on the tuned config, OOF delta vs `lgbm_pr5_tuned` ≥ **+0.0005** AND
  ≥ 4/5 folds; then re-priced at the deliverable (blend with RealMLP — which also needs the
  route features to be fair, so RealMLP gets a route variant too).
- **Prior**: ~80% that `route` clears the bar on the tuned config (each component already
  does solo on base); ~40% that `route_plus` adds ≥ +0.0002 over `route`.

- **PR-10 RESULT (2026-10-07)**: `profile` **−0.000360** 0/5 (harmful), `segment` −0.000094
  0/5, `numcat` **+0.000488 5/5** (sub-bar, but the signal that led to PR-12). Decomposition
  (base config, vs `lgbm_pr1_base`): fd only +0.000409; counts only +0.000576; cats only
  +0.000387; age −0.000467; delays −0.000239. **The whole effect is `flight_distance`.**
- **PR-12 RESULTS (2026-10-07)**. Facts: all 3,521 synthetic fd values are original values
  (100.0% of rows); per-route sat rates correlate 0.865 between synthetic and original.
  Base config, vs `lgbm_pr1_base`: `route` **+0.001416** 5/5 (0.960184); `route_plus`
  +0.001319 (finer keys dilute); on top of `route`: `route_agg` (route means of features)
  +0.000073 4/5, `route_origk` (original rate on fd+cats / fd+age / fd+3 ratings) −0.000103,
  `route_flight` (TE of fd+delays, a "same flight" key) −0.000055, fd as raw category
  −0.000894; TE smoothing 5 / 80 / auto vs 20: +0.000065 / −0.000065 / +0.000041 (noise).
  Exact-match lookups into the original (fd+age+cats etc.): residual corr ≤ 0.006 — the
  generator does not copy individual rows.
  **Tuned config (`tune_lgbm.py --evaluate --route` → `lgbm_pr12_route`): OOF 0.960522,
  +0.001520 vs `lgbm_pr5_tuned`, 5/5 → PASS.** `xgb_route` (GPU, v1 params) 0.96020
  (+0.00158 vs `xgb_v1`). Route features live in `src/route.py` (all legs take `--route`).

- **PR-11 RESULT: FAIL (dead).** RealMLP v1 seeds 42/43/44 in the submitted blend: M=2
  +0.0000085, M=3 +0.0000113 — RealMLP's internal n_ens=8 already averages. Seed 45 and the
  LGBM seeds were not run. **Closed: seed bagging of RealMLP.**
- **PR-13 — residual route encoding** (exploratory screen, then tuned): `route_residual()` in
  `run_lgbm_arm.py` — inner 5-fold route-free LGBM gives p, per-fd shrunk mean of (y − p),
  train rows encoded out-of-inner-fold. Base config, over `route`: +0.000145 (4/5); alone it
  beats the whole route set (+0.000086). Finer residual keys (fd×class/travel/cust) +0.000092
  (worse); original-data residual on top −0.000014 (null). **Tuned (`lgbm_pr13_resid`):
  0.960613, +0.000092 vs `lgbm_pr12_route`, 5/5 — sub-bar, but NOT absorbed at the blend:**
  honest blend with `realmlp_route` 0.961023 vs the submitted recipe 0.960932 (+0.00009).
  XGB/Cat route legs get weight ≈ 0 (`xgb_route` 0.96020, `catboost_route` 0.95994).

### PR-14 — LGBM re-tune on route+resid features (registered 2026-10-07, before any number)

- **Arm**: `src/tune_lgbm_route.py` — 20 TPE trials (cut from 30 for time; best-so-far saved every trial), PR-5 protocol (3-fold, 50% subsample),
  route TE + residual fit per search fold and cached; trial 0 = PR-5 params (incumbent).
  Evaluate 5-fold → `lgbm_pr13_resid_pr14`.
- **Bar (ADOPT)**: honest blend with `realmlp_route` ≥ +0.0005 over the same blend with
  `lgbm_pr13_resid`. Bookkeeping swap allowed at any positive honest delta.
- **Prior** ~5% for the bar (lesson 7, and PR-5 itself was +0.00023); ~50% for a positive
  delta, because route features changed the problem the PR-5 params were found for.

### PR-15 — RealMLP embeddings on top of route (2026-10-07)

`--fdcat` (fd as an embedded category, levels with count ≥ 20) was run as a screen first:
`realmlp_route_fdcat` **0.96092** (folds 0.96122/0.96005/0.96158/0.96107/0.96083; vs
`realmlp_route` 0.96065, +0.0003 on every fold). Honest blend with `lgbm_pr13_resid`
**0.961149** (w_mlp ≈ 0.59) vs submitted recipe 0.960932; 3-leg (+`realmlp_route`) 0.961168.
Then registered before their numbers: `--fdcat --resid`, `--fdcat --agecat`,
`--fdcat --ratingcat`. Bar (ADOPT as the NN leg): honest 2-leg blend with
`lgbm_pr13_resid` ≥ +0.0002 over the fdcat leg's 0.961149 (a lower bar than +0.0005 because
it is a leg *swap*, not an added lever, and costs nothing). Prior ~20% each: the embedding
is already the NN's route channel.

- **PR-15 results**: `--fdcat --ratingcat` 0.96091 solo, 2-leg honest 0.961133 (**FAIL**,
  −0.000016; as a 3rd leg +0.000024 = averaging); `--fdcat --resid` 0.96094 solo, 2-leg
  0.961148 (**FAIL**, −0.000001 — the embedding already carries what the residual adds to
  trees). TabM `--route` (fd embedding + route TE): fold 1 0.95853 at 33 min/fold → killed by
  timebox (far below RealMLP). Segment-aware LR stacker (travel×class×customer dummies ×
  logits) over the best 2 legs: 0.961133-0.961152 vs rank blend 0.961149 — null.

- **PR-14 RESULT: FAIL by screen, stopped at 15/20 trials.** Incumbent (PR-5 params) 0.95935
  on the search subsample; best of 15 trials 0.95936 (+0.00001). The route features did not
  move the LGBM optimum; 5-fold evaluate not run (gate on magnitude). **Closed: LGBM re-tune.**
- **PR-15 cont.**: `--fdcat --agecat` **0.96102** solo, 2-leg honest 0.961209 (+0.00006 vs
  fdcat — sub the +0.0002 swap bar, but the best NN leg); `+--delaycat` 0.96102, 2-leg
  0.961223; `--fdmin 5` (more fd levels) 0.96091, 0.961138 (null). `embedding_size 16`
  override produced bit-identical folds to the default — `train_realmlp.py` uses OUR
  `data_science_stuff.kaggle.models.RealMLP_TD_Classifier`, not pytabkit's, and unknown keys
  are silently ignored. Its keys: `embed_dim` (cat embedding size, default 8), `ls_eps`
  (label smoothing), `onehot_thresh`. A `use_ls` override was likewise a no-op (run killed).
  4-leg honest (resid LGBM + agecat + delaycat + fdcat) **0.961243** → submission 3.

- **More nulls (2026-10-07 pm)**: `route_persona` (TE of age×gender×customer×travel) +0.000051;
  `route_fdage2` (TE fd×age) −0.000069; LGBM native-categorical fd with strong cat
  regularisation −0.000195; residual k 5/50 vs 20: +0.000167/+0.000160 vs +0.000145 (flat);
  `catboost_route_fdcat` (fd, age as CatBoost cats) 0.95996, +0.000009 at the blend;
  RealMLP `--fdclasscat` 0.96101, +0.000014 as a 4th leg; row `id` carries nothing (AUC
  0.50007, no residual drift / autocorrelation).

- **RealMLP config variants** (correct keys): `ls_eps 0` 0.96106; `+embed_dim 4` **0.96108**;
  `embed_dim 2` 0.96107; `embed_dim 24` 0.96100; ls0+delaycat 0.96104; ls0+emb4+fdclass
  0.96103. All within ±0.00005 solo; together they add ~+0.00003 at the blend (averaging).

### PR-16 — original rows appended to RealMLP training folds (registered 2026-10-07, before any number)

- **Arm**: `train_realmlp.py --route --fdcat --agecat --orig` with `ls_eps 0, embed_dim 4`
  (paired against `realmlp_route_ls0_emb4_fdcat_agecat`, 0.96108): all 129,880 original rows
  appended to each fold's training split only, `is_original` = 1 kept as a feature; route TE
  for those rows is transformed by the fold's encoder (never fit on them).
- **Bar (ADOPT)**: solo OOF ≥ +0.0003 over the paired leg AND ≥ 4/5 folds positive, then a
  positive honest delta at the blend. **Prior ~10%** (PR-1 null on LGBM; delays off-
  distribution) — the only reason to run it is that RealMLP has differed from LGBM all day.
- **RESULT: FAIL on the solo bar, but real and it pays at the blend.** OOF **0.96128**
  (folds 0.96156/0.96040/0.96204/0.96136/0.96109), delta **+0.000196**, **5/5** positive
  (+22/+27/+28/+12/+5 e-5), corr 0.9993. Honest 2-leg with `lgbm_pr13_resid` **0.961391**
  (vs 0.961209 for the no-orig agecat leg). Adopted for the submission as bookkeeping (best
  honest OOF), not as a bar-clearing lever. Reading: unlike LGBM (PR-1), the NN uses the
  original rows — plausibly because they add ~34 labelled rows per route embedding.
- **LGBM + original rows, with route (bookkeeping follow-up, same session):** PR-1's null does
  NOT hold once route features exist. `lgbm_pr12_route_orig` 0.960743 (+0.000221 vs
  `lgbm_pr12_route`, 5/5); `lgbm_pr13_resid_orig` **0.960811** (+0.000289 vs
  `lgbm_pr12_route`, 5/5) — best LGBM leg. Both sub-bar solo; blend honest 0.961435.

### 2026-10-08 — pre-registrations (written before any of today's numbers)

Instrument for every deliverable read today: `src/blend_nm.py` (rank blend, Nelder-Mead weights
fit on a fixed half, scored on the other, both directions; reproduces the submitted
`blend_final_1007_orig2` at honest **0.961435**). Reference = that 5-leg blend.

- **PR-17 — original-row dose for RealMLP.** Arms: `--origrep 2` (original rows ×2),
  `--orignodelay` (original delay columns → NaN, fold-median filled). Paired vs
  `realmlp_route_ls0_emb4_fdcat_agecat_orig` (0.96128). Bar (swap): solo ≥ +0.0001 at ≥ 4/5
  folds, then a positive honest delta at the blend. Prior ~30% (×2): PR-16 says the NN uses
  the original rows; the dose-response is untested.
- **PR-18 — full-data refit (learning-curve read).** Test predictions today are the mean of 5
  models each fit on 80% of train; refitting on 100% gives every route ~25% more rows. OOF
  cannot see this, so measure the learning curve instead: tuned LGBM + route + orig, folds 1-3,
  training on 50% / 75% / 100% of the fold's synthetic rows. Fit AUC vs log2(n) and extrapolate
  to 1.25×. Bar (ADOPT a refit leg): extrapolated gain ≥ **+0.0003** for the leg. Prior ~35%
  (route-ID problems are data-hungry per key). Risk: a refit has no early-stopping set — use
  the per-fold best_iter × 1.1 (LGBM) / fixed best epoch (RealMLP).
  **Measurable variant (preferred, registered at the same time):** 10-fold CV of the main legs
  (`--nfolds 10` / `S6E10_NFOLDS=10`) — every model sees 90% of train and the OOF *measures*
  it, so no extrapolation is needed. Bar: honest blend with the k10 legs swapped in ≥ **+0.0002**
  over the reference (a swap, costs nothing but GPU/CPU time).

- **PR-19 — original-data teacher logits on top of route** (registered before its number;
  source: public notebook goodpjw2008 "More Aux Carriers" reports +0.0007 for XGB *with* route
  features, +0.0002 without — our PR-3 tested the teacher only pre-route). `src/og_model.py`:
  XGB d8 / LGBM fit on the original only → logits for every synthetic row (no synthetic labels;
  original rows get 5-fold OOF teacher logits when appended). Screen (base LGBM, vs
  `lgbm_arm_route`): `route_og`, `route_ogx`, `route_og_agg`. Bar: base-config delta ≥ +0.0003
  at ≥ 4/5 folds → build tuned LGBM + RealMLP legs with it, then honest blend ≥ **+0.0005** over
  the reference. Prior ~55% (an external measurement on the same data).
- **PR-17 ×2 arm RESULT: FAIL** — killed after 2 folds, both below the paired leg
  (0.96148 vs 0.96156, 0.96022 vs 0.96040). More original dose does not help.
- XGB + route + orig rows (`xgb_route_orig`): **0.96046** (+0.00026 vs `xgb_route`).

- **PR-19 RESULTS.** Teacher (`og_model.py`): og_xgb AUC 0.95514 on synthetic train by itself
  (original 5-fold 0.99479). Base-config screen vs `lgbm_arm_route` (0.960184): `route_og`
  **+0.000295** 5/5; `route_og_agg` (+ route profile = per-fd feature means) **+0.000399** 5/5 →
  screen PASS. Tuned (`S6E10_EXTRAS=og,agg tune_lgbm.py --evaluate --route --orig` →
  `lgbm_pr12_route_og_agg_orig`): **0.960989**, +0.000246 vs `lgbm_pr12_route_orig`, 5/5 — best
  LGBM leg. **At the deliverable: honest +0.000088** (0.961524 vs 0.961435; weight 0.32) →
  FAIL vs the +0.0005 bar (lesson 7: the RealMLP leg already carries most of it), adopted as
  bookkeeping. RealMLP + og,agg: folds 1-2 0.96144 / 0.96038 vs 0.96156 / 0.96040 — the NN does
  not use the teacher.
- **Residual-key probe (blend residuals, split-half):** no key (fd, fd×cats, age, persona,
  delays, rating tuples) carries correctable signal — every additive correction ≤ 0. kNN lookup
  of the nearest original row within the same route: residual corr ≈ 0 (no near-copying).
- **PR-21 — public OOF/test pairs (external; a separate, clearly labelled track).** Public
  datasets on the SAME folds (`StratifiedKFold(5, True, 42)`, id order — verified):
  `cdeotte/s6e10-tfm-oof-and-test-predictions` (11 TFMs) in `data/public_oof/`. LR-on-logits
  stack (`src/stack_public.py`): ours (6 legs) 0.961534 → ours + 11 TFMs **0.961783**
  (+0.00025; TabPFN-3.5 solo 0.961145 is the top coefficient). Our own TabPFN leg is blocked:
  TabPFN-3 needs the Prior Labs token/licence, not on this box.

- **PR-21 cont.** Full public pool (`cdeotte/s6e10-124-public-oof-and-test-predictions`, 124
  pairs, + the 11 TFMs; same folds, no member above 0.96175 solo → no obvious leak): public-only
  LR stack **0.962093**; + our 7 legs **0.962089-0.962101** (C 0.03-0.3) — **our legs add
  nothing to the public pool.** `submissions/hybrid_lrstack_v1.csv` (C=0.3, OOF 0.962101) is
  written but NOT submitted: using public predictions is the user's call. Our own best:
  `submissions/blend_own_1008a.csv` (7 legs, honest 0.961551).
- **RealMLP schedule re-screen (fold 1, vs base 0.96156):** flat_anneal e6 0.96151, e4 0.96139,
  e8·flat0.5 0.96157 — noise. Schedule stays closed (PR-7).
- **PR-22 — TabICL leg (registered before its number).** Open-licence TFM (`/tmp/tabicl_venv`,
  `src/export_tfm.py` → `src/run_tabicl.py`) on route TE (per fold) + og + agg + aux, stratified
  context subsample. Probe (fold 1, first 20k val rows; LGBM og_agg 0.96172 on the same rows):
  ctx 30k 0.95969, 100k 0.96032, 200k 0.96069. Full run at ctx 100k (test predicted once by
  the fold-1 model). Bar: positive honest gain at our blend AND in the public-pool stack
  (the pool is the real competitor now). Prior ~25% (public TFMs earn the largest stack
  coefficients despite weak solos).

- **Aux-task features (PR-20, `src/aux_feats.py`, label-free, train+test+original, 5-fold
  XGB per column):** tuned LGBM + og,agg,aux 0.961114 / + auxall **0.961125** (+0.00013 vs og,agg,
  5/5); LGBM `extra_trees` + lr 0.02 (`best_params_lgbm_xt.json`) **0.961156**; RealMLP + auxall
  (orig rows, no og) **0.96147** (+0.00019 vs its no-aux twin, 5/5) — best leg. Own blend after
  these: honest **0.961695** (`blend_own_1008b.csv`). XGB + og,agg 0.96087 → +0.000002 at the
  blend (GBDT copies).
- **PR-22 RESULT: FAIL.** TabICL ctx 100k: OOF folds 0.95960/0.95839/0.96030/0.96018/0.96007;
  rank-blend weight 0; LR stack +0.000004. The public TFM weights do not transfer to a
  small-context TabICL.

- **PR-25 — honest transductive pseudo-labelling (registered before its number).**
  `src/pseudo_lgbm.py`: per fold, stage 1 = LGBM-xt leg (og,agg,auxall, orig rows); stage 2 =
  refit on the same rows + all test rows with SOFT labels from the fold's own stage-1 model
  (`cross_entropy` objective), test weight 1.0 / 0.3. The val fold never enters training and
  the pseudo-labels depend only on the fold's training labels → honest OOF. Paired vs stage 1.
  Bar: OOF ≥ +0.0002 at ≥ 4/5 folds, then a positive honest delta at the blend. Prior ~20%
  (self-distillation mostly smooths; the hoped-for mechanism is more rows per route). Not in
  the public pool's tried list (they used soft labels of TRAIN rows: +0.00005).

- **More 10-08 legs.** CatBoost CPU (numeric/rating twins as categoricals + og,agg,auxall + orig
  rows; `--numcat`) **0.96094** (old CatBoost route 0.95996) → own LR stack +0.000028; CatBoost
  GPU depth 6 / l2 10 0.96108 → +0.000000 (redundant). RealMLP aux + delaycat 0.96150; + auxr
  0.96146 (null). LGBM-xt + auxr (E[fd], E[age] given the other 20 columns,
  `src/aux_reg.py`) 0.961197 (+0.00004, 3/5). **Own LR stack (`stack_public.py --ours`, 13
  legs) 0.961744** > NM blend 0.961695 → `submissions/own_lrstack_1008.csv` (not submitted).
- **PR-18 measured (10-fold):** RealMLP auxall at 10 folds **0.96157** vs 0.96147 at 5 folds
  (+0.0001 for 80→90% data); swapped into the own stack +0.000047 (0.961791). Sub-bar per leg,
  but it applies to every leg; a 100%-data refit should give about twice that on test.
- **PR-25 fold 1:** stage 1 0.96110 → pseudo-labelled (w=1) **0.96123** (+0.00013), w=0.3
  0.96117. Fold 1 took 58 min with both arms; rerunning all folds with w=1 only. RealMLP
  version queued (`--pseudo 1.0`: test labels sampled from the fold's own stage-1 model).
  **RealMLP pseudo: FAIL** — fold 1 0.96148 vs its stage 1 0.96168 (−0.0002); sampled hard
  labels add noise. Killed after fold 1.
  **PR-25 LGBM: FAIL** — w=1 per fold +0.00013 / −0.00005 / −0.00004 (folds 1-3), mean ≈ 0;
  killed after fold 3 (32-38 min/fold). Honest self-training on test adds nothing here.

### 2026-10-09

- **User directive: "objective is winning"** → the public OOF library (PR-21) is used in the
  deliverable from today; own-only blends remain the fallback/second final.
- **Pool rebuilt** (`src/pool_build.py` → `data/public_oof/_pool/`, logits, id order): every
  public source normalised (npy / npz / csv / wide parquet), checks = length, NaN, OOF AUC
  ≤ 0.9635 (in-sample guard), test alignment (Spearman ≥ 0.9 vs our LGBM test leg), near-dup
  (corr > 0.99999). **Stack-level members are excluded** (najiama blends `nj_*`, anything named
  blend/stack): they were fit on their own folds and leak into a nested stack. 199 pairs
  (was 135). Public-only LR stack 0.962100 (C 0.1; was 0.962093) — the new sources add ~0.
- **⚠️ RealMLP OOF was inflated.** `train_realmlp.py` used `epoch_select="auc"` — the epoch is
  picked on the very fold it is scored on (the PR-2 caveat, now measured). Honest fixed-epoch
  runs (new `epoch_select="last"` in the package; the schedule anneals over `epochs`), fold 1,
  same config as the best leg: e4 **0.96144**, e5 0.96141, e6 0.96130 vs the selected 16-epoch
  run ~0.9616-0.9618 ⇒ inflation **~+0.0002 on the solo**, and the LR stack hands the inflated
  legs the largest coefficients (+0.48, +0.35 in pool + ours, 0.962143). Every RealMLP number
  in this file before 10-09 is optimistic by about that much; the GBDT legs are unaffected
  (early stopping on the val fold moves them ≤ 1e-5). Many public pytabkit RealMLP members
  restore the best epoch on X_val too, so the public pool's OOF is optimistic as well (public
  LB sits ~0.0004 below public stack CVs).
  **Rule from now on: every NN leg that feeds a stack uses `epoch_select="last"`.**
- **Honest RealMLP legs (k10, e4, `epoch_select="last"`):** auxall **0.96144**, +delaycat 0.96145,
  +fdclasscat 0.96145, seed 43 0.96139, og,agg 0.96128 (vs 0.96157-0.96160 selected).
- **Stacks (LR on logits, nested on the base folds, `stack_public.py`):** pool-only (199) 0.962144
  (C 0.1; C 0.001 / 0.003 / 0.01 / 1 / 3: 0.962039 / 0.962090 / 0.962136 / 0.962144 / 0.962141);
  pool + 6 own legs (2 honest RealMLP + CatBoost + 3 LGBM, all k10) **0.962162** (C 0.3) →
  **SUBMITTED `hybrid_lrstack_1009.csv`**. Our legs add +0.00002 to the pool — noise level.
  + all 5 honest RealMLPs: 0.962152 (no gain from NN breadth).
- **Inflation hedge:** pool without the 53 public MLP-type members (realmlp/rmlp/mlp/deepfm/gl_e/gl_n)
  + our 6 legs **0.962107** (−0.000056 OOF — within the plausible inflation of those members) →
  **SUBMITTED `hybrid_lrstack_1009_nomlp.csv`** as the robust candidate. With 5 honest RealMLPs 0.962104.
- **Nulls (10-09):** nested residual LGBM on top of the LR stack with raw context + top-20 members
  (`stack_resid.py`) 0.962134 vs 0.962144 (−0.00001, 2/5 folds up); GPU kNN leg on the TFM
  matrices (`knn_leg.py`, |Spearman|-weighted metric, solo 0.9577 at k 400) → stack −0.000013;
  alternative original-data teachers (LGBM depth 2-8, min_child 500/2000, no delays, no fd) all
  0.9534-0.9552 solo vs og_xgb 0.9551 — teacher structure does not matter.
- **Afternoon:** honest e3 leg 0.96138 (< e4); TabICL route leg in the full pool −0.000013 (closed);
  per-member test/OOF logit-scale check: all 0.995-0.999 (no broken public member). `--fullfit`
  added to `train_realmlp.py` (fixed epochs, all train rows): `hybrid_lrstack_1009_fullmix.csv` (test of
  the 2 RealMLP legs = mean logit of k10 + 2 full-fit seeds) is Spearman 0.99998 with the submitted
  file — not worth a slot. 15 more pairs pulled from public notebook OUTPUTS (kratosyan route-id
  teacher, kagankoral encodings, koumeimaki ladder, hermengardo CatBoost, amanatar original-prior):
  full stack unchanged (0.962161), no-MLP hedge +0.00003 (0.962135).
- **Evening re-pull (18:45):** new since 09:45 UTC — dmitriykolmikov / rabari9999 datasets (submissions
  only), 5 notebook outputs; only chinzorigtganbat's v10 has OOF/test pairs (4 GBDT members, `cz_*`;
  its blend columns are stack-level and excluded). Pool 218: full stack **0.962173** (C 0.1), no-MLP
  0.962131 — both below the pre-set submit bars (0.962190 / 0.962165) → **not submitted**.
- **TabPFN: blocked.** The cached Prior Labs token had expired; tabpfn 9.1 (default checkpoint is now
  TabPFN-3.5) deleted it after verification failed. The user must log in at ux.priorlabs.ai, accept the
  v3.5 license and export `TABPFN_TOKEN`; then `/tmp/tabpfn_venv/bin/python src/run_tabpfn.py` (venv
  must be recreated after a reboot: `uv venv /tmp/tabpfn_venv -p 3.12 && uv pip install tabpfn`).

### 2026-10-10

- **Pool re-pull:** added koumeimaki `s6e10-route-tfm-nn-oof` (full-context TabPFN-3.5 0.96118,
  TabICL, TabM, RealMLP, 3 GBDTs; same folds, verified). `pool_build.py` now clears `_pool/`
  first (members dropped by dedup were lingering: 378 pairs on disk vs 225 built). Pool 225 +
  6 own legs: **0.962162** (C 0.1) — flat; the pool is saturated by more of the same kinds.
- **TabPFN-3 runs locally without a licence click**: `run_tabpfn.py` defaults to the cached
  v3 checkpoint (`~/.cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt`). The 3.5 checkpoint
  still needs the licence accepted at ux.priorlabs.ai (token saved at `~/.config/tabpfn/token`,
  outside the repo). Fold-1 probe (20k val rows, n_est 1): our 69 engineered columns ctx 50k
  0.96033 / **ctx 100k 0.96098**; raw 21 (fd categorical) ctx 100k 0.95871.
- **Member screen (`src/screen_member.py`)**: a candidate's fold-1 predictions vs the nested
  stack OOF, 2-feature LR, 10-fold × 5 on the 140k rows. Controls (calibrated before any
  candidate number): noise −0.000017, kNN −0.000017, TabICL −0.000015, in-pool tabfm
  −0.000021 (all ±0.000005) ⇒ the null sits at about −0.00002.
- **PR-27 (registered before its number) — TabPFN-3 on engineered features / route-local
  context.** Bar for a full 5-fold run: screen delta ≥ **+0.00002** (≥ 4σ above the null).
  Bar to submit: full stack ≥ **+0.00003** over 0.962162. Prior ~25% (TabICL on the same
  matrices was null; TabPFN is stronger and the public TabPFNs earn large coefficients).
- **PR-27 RESULTS.** TabPFN-3, 69 engineered columns, ctx 100k, full fold 1: solo 0.96008, corr
  0.9935 with the stack, screen **−0.000019 → FAIL** (= null). Route-local context (`--local 0.5`:
  each 4k-query batch's context = training rows on its routes + random fill, ctx 50k) lifts the
  solo +0.0006 over a random 50k context (0.96092 vs 0.96033 on 20k rows) but screens no better
  than random ctx 100k (−0.00008 vs −0.00011 at n=20k, where the null is wider) → closed.
  TabDPT (`src/run_tabdpt.py`, retrieval TFM): no flash attention on the 2060 → >25 min per 20k
  rows, infeasible. Naive Bayes over all columns (+ rating×segment pairs): solo 0.9335, screen
  −0.000004 → null.
- TabPFN-3 on the RAW 21 columns (fd categorical; the kind of member that earns the pool's
  largest weights — tabfm, limix2), ctx 100k, full fold 1: solo 0.95723, screen **−0.000023** →
  null. Original-data teacher logits as members: −0.000009 / −0.000012 → null.
- **PR-28 — swap-noise DAE (label-free, train + test) → MLP head (`src/dae_leg.py`)**, registered
  with the PR-27 bar. 12 DAE epochs: solo 0.95947, screen −0.000026; 60 epochs: solo 0.96015,
  corr 0.9928, screen −0.000021 → **FAIL**. A better solo just raises the correlation with the stack.
- **Segment-gated LR stack** (separate weights per class / travel / customer / class×travel):
  0.961946-0.962132, all below the single LR (0.962162).
- **PM re-pull (13:40):** `sachith7` stack-OOF update adds `lgb_cpu_strong` (0.961083) → pool 226,
  stack **0.962169** (+0.000007, below bar). Other new notebooks: submissions or weak members only.
- **PR-29 — Google TabFM on our compact features (registered 14:05, before the full number).**
  `src/run_tabfm.py` (NVIDIA `structured-data-models`, `/tmp/sdm_venv`; 1.64B params → fp32 does not
  fit 6 GB, fp16 gives NaN, **bf16 works**; ctx ≤ 3000 with 28 columns). Features: raw 21 + 4 route
  TEs + og_xgb + orig_fd_rate + fd count; 4 bags × fresh 3k context, mean logit. Probe (20k fold-1
  rows): solo 0.95925, corr 0.985, screen **+0.000084** vs TabPFN-3 −0.000109 on the same rows ⇒
  ~+0.0002 above the null — first candidate today above it (the pool's raw-column TabFM already
  earns the largest coefficient). Full run: OOF all folds, test from fold 1's 4 contexts. Bar to
  submit: full stack ≥ **+0.00003** over 0.962169. Prior ~50%.
  **RESULT: FAIL.** Full fold 1 (140k rows, same seeds): solo 0.95744, screen **−0.000022** (null).
  The 20k probe was a lucky subset — the other six 20k slices of the SAME predictions screen
  −0.00010 to −0.00014 (one +0.000018). ⚠️ **Screen on full folds only**: the reported ± is CV noise
  within the subset, not row-sampling noise, so 20k-row screens can mislead by 2e-4. Run killed
  after fold 1.
- **Kumo-large** (same runner, `--model kumo`, ctx 5000 × 2 bags, full fold 1, ~3 min/fold): solo
  0.95874, screen **−0.000017** → null. The TFM family (TabPFN-3, TabICL, TabFM, Kumo) is closed on
  this hardware (raw-column TabFM, 6 × 3k contexts — a second draw of the pool's top-weight member —
  solo 0.95604, screen −0.000022: also null); new members have to come from the public pool (re-pull 17:00 and 19:30).
- **Out-of-the-box round (15:30-16:05):**
  - Hidden-ID screen: every column's synthetic values ⊂ original values (delays: 186 of 466) — the
    generator treated all columns as categorical; no new high-cardinality ID besides fd (delaycat and
    flight keys already tested).
  - Route-reliability-gated stack (`src/stack_gated.py`, member logits × standardised log route
    count): 0.962145 / lean 3-feature gate 0.962155 vs 0.962169 → null.
  - AUC-direct stacker (`src/stack_auc.py`, LR weights fine-tuned on a soft pairwise AUC with L2 pull
    to LR): 0.961836-0.961963 → worse; log-loss LR is already AUC-optimal here.
  - **PR-30 label-noise cleaning (`src/label_clean.py`, fold 1, honest inner 3-fold p):** LGBM on the
    69 exported columns plain 0.96037 → down-weight |y−p|>0.8 (4.5% of rows) **0.96075** → soft labels
    0.7·y+0.3·p **0.96087 (+0.0005 solo)** — a real LEG gain, but screens −0.000017/−0.000018 at the
    stack (absorbed, lesson 7). Worth using for any own-only blend.
- **17:00 re-pull:** dmitriykolmikov all-categorical CatBoost (0.96067) + fast LGBM, denpugovkin exp022
  members (numeric-key CatBoost CTR, LGBM view B, RealMLP view B, 2 controls; blends excluded) →
  `data/public_oof/s6e10-pm1010-new/`, pool 233: stack **0.962154** (−0.000015) → not submitted.
- **PR-31 — soft labels / distillation into RealMLP (fold 1, honest e4 reference 0.96144).**
  `train_realmlp.py --soft a [--soft-teacher OOF | --soft-inner-nn]`: exact soft CE via weighted row
  duplication. Students (a = 0.3): inner LGBM teacher (0.9587) 0.96145; honest inner RealMLP teacher
  (0.9607) 0.96134; honest inner 5-fold LGBM on 69 cols (`src/inner_teacher.py`, 0.96052) 0.96151
  (screen −0.000015); **pool-stack OOF teacher (0.9621) 0.96197, screen +0.000032**; own k10 LGBM OOF
  teacher (0.9613) 0.96168 (screen −0.000006); kNN-400 OOF teacher (0.9576) 0.96117.
  ⚠️ The stack/k10 teachers are LEAKY for a high-capacity student: their OOF on fold-1 training rows
  comes from models that saw fold-1 val labels. The kNN placebo shows no local leak, but strength
  and leak are confounded. **Matched test settles it — LEAK:** same LGBM config/columns, teacher =
  ordinary 5-fold OOF (folds 2-5 trained WITH fold-1 val rows) 0.96068 vs honest inner 0.96052 →
  students **0.96204 vs 0.96151** (+0.00053 at equal teacher strength; data size explains ~+0.00005).
  ⇒ Distilling OOF predictions into a high-capacity student leaks the val labels through the
  teacher's models; the stack-teacher +0.000032 is leak. PR-31 **FAIL** (honest soft labels = null). LGBM soft-label sweep: a = 0.15 0.96058 < a = 0.3 0.96087.
- **Robust-combiner round (18:00-18:20):** L1 LR stack (saga) C 0.003 / 0.01: 0.962066 / 0.962083
  (worse). Monotone-in-ratings LGBM (fold 1): 0.96033, screen −0.000020 (null). OOF-vs-test
  correlation gap as an inflation detector: separates our inflated RealMLP twins from the honest
  ones by only ~1 MAD of the cross-family spread → not usable to filter public members.
- **PR-32 — member-bagged LR stack (`src/stack_bagged.py`), registered 18:20 after an 8-bag pilot**
  (pilot, 239 members: 0.962183 vs single LR 0.962154 on the same members). Each bag = nested LR
  (C 0.1) on a random half of the members; deliverable = mean of z-scored bag logits. Settings fixed
  before the full run: 32 bags, frac 0.5, seed 0. Bar to submit: **≥ 0.962199** (+0.00003 over
  0.962169, the best single-LR stack today). Mechanism: variance reduction of the stacker AND no
  single (possibly inflated) member can dominate — a robustness gain the LB should also see.
  **RESULT: FAIL vs bar.** 32 bags 0.962180 (1/4/8/16 bags: 0.962110/0.962168/0.962183/0.962182);
  excluding the 17:00 members 0.962182; frac 0.3 0.962130. Real but small (+0.00001-0.00003);
  written (`bagstack_1010*.csv`), not submitted.
- **Residual MLP stacker** (`src/stack_mlp.py`, MLP on member logits added to a frozen LR term,
  zero-init head): 0.962150 vs LR 0.962156, every fold ±0.000002 → null; LR is the ceiling.
- **Combiner variants (`src/stack_variants.py`)**: + consensus curvature / member spread 0.962162
  (= base); rank-logit inputs 0.961148 (worse). LR on logits stays.
- **Inflation transfer measured:** swapping our two honest RealMLPs for their epoch-selected twins
  (+0.00014/+0.00016 solo) moves the stack +0.00002 (0.962182). One inflated member passes ~1/7 of
  its inflation through; the 53 public MLPs can plausibly explain all of the hedge's −0.000055.

## Next steps (remove each item when implemented)

**From 10-09 (do these first):** (a) RESOLVED — public pool is used (user: "objective is
winning"). The work is "find members that move the pool": honest NN breadth, kNN, residual
stacker and teacher variants all failed today; the only member class with large stack weights
we lack is a TabPFN-3.5 leg (needs the user's Prior Labs license, see 10-09). (a2) Finals:
`hybrid_lrstack_1009.csv` (0.962162) vs the inflation hedge `_nomlp` (0.962107) — decide with
`select-finals`; re-pull the public pool before the end (new members keep appearing). (b) Final-submission refit: legs on 10 folds (90% data) or
a 100% refit for the test predictions (PR-18 learning curve: ~+0.0008 AUC per data doubling
for LGBM ⇒ ~+0.00015 for 80→90%) — do it once the leg set is frozen. (c) TFMs earn the
largest pool weights; ours (TabICL ctx 100k) did not — a TabPFN leg needs the Prior Labs token.
Done 10-08 from the 10-07 list: orig ×2 (worse), orig without delays (flat, killed), XGB/Cat
with route+orig (XGB +0.00026 solo, 0 at the blend); LGBM re-tune with orig not run (PR-14
showed the optimum did not move).

Closed so far: original as rows (PR-1) and as features (PR-3); pairwise rating TE (PR-4a);
categorical ratings (PR-4b); duplicate/frequency structure (PR-6, by screen); RealMLP tuning (PR-7); TabM (PR-8); LR stack (PR-9). Sub-bar but real: RealMLP (PR-2), LGBM tuning (PR-5).
**Untested lever classes:**
1. Tune XGB / CatBoost — but PR-9 says the combiner gives the GBDTs little weight; price at the
   blend before spending GPU hours.
2. Information levers not yet tried: per-row rating-profile features (z-scores within row,
   dispersion, rank of each rating within the row), class/travel-type × rating interactions,
   kNN-distance features. PR-4 suggests trees already resolve pairwise structure, so screen
   cheaply first.
3. Multi-seed bagging of the two submitted legs — variance reduction; screen by the M-slope
   (lesson 1) before trusting any per-seed gain.

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
