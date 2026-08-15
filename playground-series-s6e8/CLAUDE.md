# Predicting Smartphone Addiction (playground-series-s6e8)

**Task**: binary classification — predict `addicted_label` (1 = flagged as smartphone-addicted)
**Metric**: ROC AUC (probability of the positive class)
**Deadline**: the Kaggle API returned the literal string **`2026-08-31 23:59:00`** with no
timezone field — **queried from the source**,
`competitions_list(search="playground-series-s6e8").deadline` (and `.evaluation_metric`), on
**2026-08-04**. Kaggle's convention is UTC, so treat it as 2026-08-31 23:59 UTC, but that tz is
convention, not something the API stated. Competition opened 2026-08-01; **10 submissions/day**,
max team size 3.
Never carry a deadline that no document sourced: an unsourced "~Aug 2" ran through 22 days of
s6e7 planning and the real date was Jul 31, which deleted the endgame margin the whole
protocol assumed.

---

## ⚠️ CRITICAL: Row-order invariant — always load labels and IDs through build_features()

`features.py::build_features()` sorts every dataframe by `["id"]`. All `results/oof_<model>.npy`
and `results/test_<model>.npy` arrays are stored in this sorted order.

The sort is currently a no-op (the shipped CSVs are already id-ascending) — that is exactly why
it is dangerous. Do not "optimize it away", and do not assume raw CSV order is safe just
because it happens to match today.

Any script that loads `y` or `test_ids` and combines them with the npy arrays **must** go
through `build_features()`, or predictions silently misalign (→ ~0.5 AUC submission):

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — raw CSV order ≠ npy order
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

---

## ⚠️ No data leakage — fold-aware rules

Violating these produces inflated OOF scores that do not transfer to the leaderboard.

**The v1 baseline uses no target-derived features at all** (raw columns + pure-arithmetic
ratios, no imputation, no encoding fit on `y`), so 0.96312 has no leakage surface and is safe
as the reference number. Everything below is forward-looking guidance for legs 2+.

1. **Target encoding is fold-aware, NEVER global.** Fit any `TargetEncoder` / smoothed
   target-rate feature on the fold's training split only, then `transform` the val split:
   ```python
   # CORRECT                                  # WRONG — leaks val targets
   te.fit_transform(X_tr[["c"]], y_tr)        # te.fit(X[["c"]], y)
   te.transform(X_val[["c"]])
   ```
   Use `data_science_stuff.kaggle.encoding` rather than hand-rolling it.
2. **Imputation, scalers and quantile bins are fold-aware too.** With 4–20% missing in every
   column, imputer statistics are a real leakage surface here — fit inside `fit_fold`.
3. **Test features must be computable without the target.** There is no time/group structure in
   this dataset, so there are no lag features to get wrong — but no per-row target either.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 691,369 | 14 (incl. `id` + target) |
| test  | 296,302 | 13 |

**Target balance**: `addicted_label` mean = **0.7094** — majority-positive, mildly imbalanced.
(The sample submission is the constant 0.709424, i.e. the train base rate.)

**Missing values: everywhere.** Every one of the 12 feature columns has 4–20% nulls; only 39%
of train rows are complete. Mean nulls/row: train 1.259, test 1.273.

### Raw features + null rates

| Column | Type | train null % | test null % | Δ |
|---|---|---|---|---|
| `age` | float (18–35, 19 distinct) | 4.18 | 5.78 | **+1.60** |
| `daily_screen_time_hours` | float (0.5–15.0) | 13.86 | 11.07 | **−2.79** |
| `social_media_hours` | float (0.0–8.0) | 19.38 | 16.00 | **−3.38** |
| `gaming_hours` | float | 18.34 | 20.05 | **+1.71** |
| `work_study_hours` | float | 7.45 | 9.37 | **+1.92** |
| `sleep_hours` | float (4.5–9.0) | 6.43 | 7.58 | +1.15 |
| `notifications_per_day` | float (20–250) | 9.78 | 11.55 | +1.77 |
| `app_opens_per_day` | float (15–180) | 11.67 | 8.68 | **−2.99** |
| `weekend_screen_time` | float (0.51–17.56) | 16.21 | 17.11 | +0.90 |
| `gender` | str: Male / Female / Other | 4.20 | 4.80 | +0.60 |
| `stress_level` | str: Low / Medium / High | 7.98 | 6.62 | −1.36 |
| `academic_work_impact` | str: Yes / No | 6.40 | 8.68 | **+2.28** |

Categoricals are near-uniform in train (gender 223.7k/221.6k/217.1k;
stress 220.9k High/207.8k Low/207.6k Medium; impact 330.6k Yes/316.6k No). No unseen levels.

---

## Key EDA findings

### The budget identity (the most useful fact in the dataset)

**`social_media_hours + gaming_hours + work_study_hours ≤ daily_screen_time_hours` holds in
100.0000% of complete train rows**, and the residual attains exactly 0 (min −0.0000, q01 0.0900,
median 0.750, max 11.53). So the generating process is

```
daily = social + gaming + work + other,     other ≥ 0
```

Two consequences that a GBDT cannot reach on its own:

1. Every term is non-negative, so the **partial** sum of whichever parts are observed is still a
   valid lower bound on `daily` — including on the 13.9% of rows where `daily` itself is missing.
   `sgw_lb` in `features_v2.py` is therefore an always-computable (0 nulls) lower bound on the
   single most informative column in the dataset.
2. Two v1 features were mis-specified: `work_study_hours` is *inside* `daily`, not disjoint from
   it, so `free_hours = 24 − sleep − work − daily` double-subtracted work, and
   `other_screen_hours` failed to subtract it. Corrected in `features_v2.py`.

### Missingness is block-structured, not complementary

Nulls **positively co-occur** — all 66 column pairs have lift > 1 (observed joint null ÷
independent joint). Nulls-per-row sd is 1.396 vs 1.047 under independence. There are four blocks:

| Block | Lift (train) |
|---|---|
| `age` ~ `gender` | 3.61 |
| `sleep_hours` ~ `stress_level` ~ `academic_work_impact` | 3.21–3.27 |
| `notifications_per_day` ~ `app_opens_per_day` | 3.17 |
| `daily` ~ `social_media` ~ `gaming` ~ `weekend` (the screen block) | 2.19–2.22 |
| background (every other pair) | ~1.32 |

Practical consequence: when `daily` is missing, its screen-block siblings are *more* likely to be
missing too — but at least one sibling is still available in **93.2%** of those rows.

`weekend_screen_time` is **not** a weekly aggregate: it is a rate on the same scale as `daily`
(median ratio 1.24, weekend ≥ daily in 85.6% of rows). Do not divide it by 7.

### Signal is concentrated in three columns, and they are the imputable ones

| Column | single-feature AUC | OLS R² from siblings |
|---|---|---|
| `daily_screen_time_hours` | **0.889** | **0.807** |
| `weekend_screen_time` | **0.880** | **0.640** |
| `social_media_hours` | **0.858** | 0.482 |
| `work_study_hours` | 0.653 | 0.419 |
| `gaming_hours` | 0.621 | 0.312 |
| `age`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day` | 0.49–0.54 | 0.000–0.005 |
| `gender`, `stress_level`, `academic_work_impact` | ~0.50 (levels 0.705–0.721) | — |

The bottom two groups are noise **and** unreconstructable — never impute them; it can only add
noise. All three categoricals are essentially uninformative.

### The label is a noisy sigmoid of a latent index, not a hard rule

Logistic regression on the 9 raw numerics reaches **AUC 0.9277** (vs LGBM 0.96312), with score
ventiles running 0.09 → 1.00 monotonically and no discontinuity. Standardised coefficients:
`social +1.28`, `daily +1.13`, `weekend +1.13`, `work −0.37`, `gaming −0.31`, everything else
≈ 0. Substituting the budget identity, that index is equivalent to **`2·social + other +
weekend`** — social media counts double; gaming and work screen time do not count at all.

### Missingness carries no marginal signal about the target

Every `is_null(col)` has AUC **0.4999–0.5010** against `addicted_label`; `n_missing` is 0.5017.
Mask indicators are dead **as marginal features**. This does *not* close mask × value
interactions, and it does not mean masks are structureless — see the MAR AUCs below.

### Device: LightGBM CPU beats CUDA here, and the two paths disagree

One fold (553k × 21): **`cuda` 53.8 s, AUC 0.96247, best_iter 903** vs **`cpu` (n_jobs=12) 22.2 s,
AUC 0.96217, best_iter 608**. CPU is 2.4× faster *and* the 0.0003 device gap is the same size as
the effects being measured — so **every arm in a paired comparison must use the same device**.
The sweep is locked to CPU in `run_arm.py`.

---

## Current best

**`bag(xgbt×6)` = 0.96609 — tuned XGBoost on `s1`, six seeds bagged. Built 2026-08-15.**
Params adopted 2026-08-13 at +0.001139 paired over untuned.

| Leg / blend | OOF AUC | Note |
|---|---|---|
| `s1` single seed | 0.96371 | the number this file used to call "current best" |
| **`bag(lgbm×6)`** | **0.964346** | ← the null the Day-6/Day-10 breadth reads were priced against |
| `bag(xgb×6)` | 0.965066 | untuned; strongest family before tuning |
| Day-10 four-family blend | 0.965090 | sixteen models, Nelder-Mead |
| `xgbt_s1` (1 tuned seed) | 0.965668 | beats the whole blend by +0.000578 |
| `xgbt_s1` (6-seed mean) | 0.9656518 | seeds 42–47, sd ≈ 0.00003 |
| **`bag(xgbt×6)`** | **0.96609** | ← **the current best and the new null.** `submissions/bag_xgbt_s1x6.csv` |
| `bag(cat×3)` | 0.962516 | earns **weight 0.0** |
| `mlp_s4` (1 seed) | 0.941983 | earns **weight 0.0**, harmful at every weight |

**The null for every future read is `bag(xgbt×6)` = 0.96609, not `bag(lgbm×6)` = 0.964346.**
Anything priced against the old null is credited with +0.001744 it did not earn.

Bagging the tuned leg was worth **+0.000438** over the 6-seed mean (0.9656518 → 0.96609) — much
less than LGBM's +0.000635, as expected: `xgbt`'s seed-to-seed sd is ≈0.00003 against LGBM's
≈0.000043, so there is less seed variance available to average away.

**Quote the bag, not the single seed.** Seed-bagging six `s1` seeds is worth **+0.000635** over one
seed — larger than the entire +0.0003 breadth bar. Pricing anything against 0.96371 credits it
with variance reduction that is free (root CLAUDE.md lesson #1). Bagging is also *exhausted*:
the increments halve cleanly (M=2 +0.000379, 3 +0.000127, 4 +0.000064, 5 +0.000039, 6 +0.000026),
so M→∞ is ≈0.96444 and more LGBM seeds cannot reach +0.0003.

| Surface | OOF AUC (3-seed mean) | Δ vs `s0` | Script |
|---|---|---|---|
| `s0` (v1, 21 feat) | 0.96319 | — | `run_arm.py --arm s0` |
| `s1` (+budget features, 27 feat) | **0.96373** | **+0.00055 ± 0.00011** | `run_arm.py --arm s1` |
| `lgbm_v1` (CUDA, superseded) | 0.96312 | — | `src/baseline.py` |

**All OOF numbers here are mildly optimistic in the same direction**: every leg early-stops on the
fold it then writes as its OOF prediction (LightGBM `early_stopping(50)`, XGBoost
`early_stopping_rounds=100`). The bias is shared, so *orderings* are safe, but CV↔LB calibration
should be expected to read high. XGBoost gets the longer leash of the two (3000 trees / patience
100 vs 1000 / 50), so a sliver of its advantage over LightGBM is stopping-rule optimism.

**Two different noise scales — do not confuse them:**

- *Fold* spread within one run is 0.96247–0.96381 (sd ≈ 0.0005). This is the spread of per-fold
  AUCs and is **not** the resolution of a CV comparison.
- *Seed-to-seed* sd of the full **OOF** AUC is only **≈0.0001**, because OOF pools all 691k rows.
  A 3-seed paired delta therefore resolves ~0.0001.

So candidates far below the +0.0010 adoption bar are easily *detectable*. That is precisely why
the bar is set on **deployability**, not detectability, and why anything in the +0.0003…+0.0010
band goes through `screen_surface.py` before adoption — a variance reducer is significantly
positive at every seed by construction (root CLAUDE.md lesson #1).

Not yet measured: LB score (user watches the LB; work LB-blind), CV↔LB calibration.


---

## Feature engineering

All features implemented in `src/features.py::build_features()`.

**v1 is deliberately lean** — raw columns plus 9 domain ratios: `social_share`, `gaming_share`,
`other_screen_hours`, `weekend_ratio`, `notifications_per_open`, `minutes_per_open`,
`screen_share_of_awake`, `free_hours`, `stress_ord`. No imputation and **no missingness
indicators**: LightGBM handles NaN natively, and mask-derived features are a separate
pre-registered experiment. Adding them to `build_features()` later changes the surface every
OOF array lives on (root CLAUDE.md lesson #9) — if you do it, do it as a new named run and only
stack same-surface arrays.

### Surfaces

Named in `src/run_arm.py::ARMS`. **Arrays from different surfaces must never be blended or
stacked** — every one has length `len(y)`, so no shape check can catch a mix; `ensemble.py`
asserts the arm suffix instead.

| Arm | Surface | Built by |
|---|---|---|
| `s0` | v1 reference, NaN native (21 features) | `features.build_features` |
| `s1` | + budget-constraint features, still NaN native (27) | `features_v2.build_features_v2` |
| `s2` | `s1` + 12 mask indicators + `n_missing` — the **control** for `s3`/`s4` | `impute.add_masks` |
| `s3` | `s1` + fold-fit median/mode imputation + masks | `impute.median_impute` |
| `s4` | `s1` + constraint-aware relational imputation + masks | `impute.relational_impute` |
| `s5` | `s1` + pooled latent screen-intensity estimate + masks | `impute.latent_impute` |

`s2` exists so an imputation delta isn't confounded with *losing* the native-NaN signal. Since
masks turned out marginally worthless (AUC ≈ 0.500), `s2` is expected to sit on top of `s1`.

`features.py` is **frozen** — `oof_lgbm_v1.npy` lives on its surface. `features_v2.py` extends it
by calling `derive_v1`, so the `["id"]` sort has exactly one owner. `derive_v1`/`derive_v2` do
*not* sort, which is what lets `impute.py` recompute the derived columns on a fold row subset
without disturbing row order.

**Why imputation is worth doing even if it is flat for GBDTs**: an MLP cannot consume NaN at
all. `s3`/`s4` are the prerequisite for `train_mlp.py`, and a neural leg is the one inductive
bias a blend of three GBDTs cannot manufacture for itself.

### What didn't help

| Idea | Δ AUC | Why |
|---|---|---|
| *(none yet)* | | |

---

## Modelling notes

- **Shared utilities live in `data_science_stuff.kaggle`** (io / device / cv / blending /
  decision / encoding / stacking / screening / shift / models) — import, never copy,
  season-specific scripts.
- **CV**: `run_cv` from `data_science_stuff.kaggle.cv` (5-fold stratified, `random_state=42`).
  Keep the seed so OOF folds stay aligned across models — stacking depends on it.
- **Nothing in `results/` or `submissions/` is committed** except `best_params*.json`:
  `.gitignore` ignores `*.csv` and `*.npy` repo-wide, so `results/cv_scores.csv`,
  the `oof_*/test_*.npy` arrays and every submission are local-only (same as s6e5/s6e6).
  `save_cv_result` still appends to `results/cv_scores.csv` for local use, but **the durable
  score record is the experiments log at the bottom of this file** — write every run there.
- **OOF/test arrays are 1-D positive-class probabilities** (`n_outputs=1`). AUC is rank-based,
  so **there is no threshold/decision layer** in this competition — do not port s6e6's
  `optimize_thresholds` machinery here; it is a balanced-accuracy artifact.
- **Blending**: `optimize_blend_weights` operates on probabilities. For AUC, also try blending
  on **ranks or logits** — a plain probability average is not scale-invariant and often
  underperforms rank-averaging when legs are differently calibrated.
- **GPU**: LightGBM device via `src/lgbm_device.py` (`device_type="cuda"` probe, CPU fallback;
  never OpenCL `device="gpu"`). CatBoost on this 6GB box needs `max_ctr_complexity=1` and no
  `eval_set` or it falls back to CPU.
- **GPU memory rule (all PyTorch scripts)**: `del` model + optimizer + tensors before returning
  from each fold; `torch.cuda.empty_cache()` after each fold/trial in the outer loop.
- **Categoricals**: pandas `category` dtype for LGBM/XGB with the category set **unioned across
  train and test** (`baseline.py` does this) so codes match; `cat_features` for CatBoost; OHE
  for nets. Nulls in the string columns are a genuine level — do not fill them with `"missing"`
  without measuring it.
- **691k × ~21 features on CPU** — this is big enough that iteration speed matters. Parallelize
  the measurement loop before queueing runs. One 5-fold LightGBM run is ~4 min at `n_jobs=12`;
  budget arms accordingly. GPU work (`train_mlp.py`) can run concurrently with a CPU sweep.

### Scripts

| Script | Purpose |
|---|---|
| `bench_device.py` | one-fold cuda-vs-cpu timing; run before any paired sweep |
| `mask_mechanism.py` | the pre-registered mechanism test (closed, FAIL) |
| `probe_latent_index.py` | the generative-formula probe (closed, do-not-adopt) |
| `run_arm.py` | LightGBM arm sweep; `--arm`/`--seeds`, and `--compare` for paired deltas |
| `legs.py` | generic CV runner for non-LightGBM legs on a chosen surface |
| `train_xgboost.py` / `train_catboost.py` / `train_mlp.py` | breadth legs |
| `tune_lgbm.py` | Optuna, 3-fold for ranking; writes `results/best_params.json` |
| `ensemble.py` | prob/logit/rank blends, same-surface assertion, writes a submission |
| `verify_submission.py` | elementwise id check vs `sample_submission` + `.npy` cross-check |

---

## Next steps (remove each item when implemented)

1. ~~Run `src/baseline.py` for the reference OOF AUC.~~ **Done 2026-08-04 → 0.96312.**
2. ~~**Pre-registered day-1 experiment — the missingness mechanism.**~~
   **CLOSED 2026-08-04 → FAIL against the unmoved bar. Mechanism declared the SAME.**
   Prior was stated as null before running, and the null held. Reproduce with
   `python src/mask_mechanism.py`; report at `results/mask_mechanism_report.csv`.

   | Instrument | Bar | Result | Verdict |
   |---|---|---|---|
   | `mask_mechanism_report` per-column MAR-AUC gap | ≥0.02 on **≥2** columns | ≥0.02 on **1** column (`notifications_per_day`, 0.0284) | fail |
   | `adversarial_validation` on the mask matrix | ≥0.60 | **0.5649** | fail |
   | `unseen_mask_patterns` | (context) | 380 test rows, 0.128% | negligible |

   What this closes: **mask-repair** work (re-masking train to emulate a test mechanism). That
   avenue is shut.

   What it does **not** close: mask features, mask × value interactions, or imputation.

   Two side-findings worth keeping:
   - The MAR AUCs are **0.56–0.76, not 0.50** — a column's mask *is* strongly predictable from
     the other masks (this is the block structure above). The rule is simply the *same* in train
     and test. Strong within-split structure, no cross-split difference.
   - Mask-matrix adversarial AUC 0.5649 is below the 0.60 threshold at which root CLAUDE.md
     lesson #6 says OOF stops being able to *rank* sub-noise candidates. OOF ordering is
     trustworthy here — unlike s6e7 at 0.65.
3. ~~**Missingness as signal**~~ — **marginally CLOSED 2026-08-04.** Every
   `roc_auc_score(y, is_null(col))` lands in 0.4999–0.5010 and `n_missing` at 0.5017. Masks carry
   no marginal signal. They are still carried in arms `s2`–`s4` as the *control* (so an
   imputation delta isn't confounded with losing the NaN), and mask × value interactions remain
   untested.

4. ~~**Latent-index / generative-formula probe**~~ — **CLOSED 2026-08-04 → DO NOT ADOPT.**
   `python src/probe_latent_index.py`. Bar was: a 1-D index with complete-row AUC ≥0.95.

   The coefficient algebra was *confirmed exactly*: `social + daily + weekend − gaming − work`
   and `2·social + other + weekend` both score **0.9238**, i.e. the budget substitution is an
   identity, not an approximation. A grid search over `a·social + b·other + c·weekend` peaked at
   **0.9264** at `(a, b, c) = (2.75, 0.75, 1.00)`.

   Both are **below the 0.9277 fitted-linear ceiling** and far below the 0.95 bar. The label is
   not a recoverable low-dimensional formula; the 0.9277 → 0.96312 gap is genuine interaction
   structure that no 1-D index captures. Adding the best index to the raw set moved an
   *in-sample* LGBM by +0.00029 — and in-sample overstates, so the real effect is smaller than
   the 0.0005 noise floor.

   This closes the *single-index* form of the question. It does not close interaction features.

   Kept for interpretation: the fitted weights say social-media hours carry ~3.7× the weight of
   unclassified screen time, and gaming/work screen time carry ~none. That is a real description
   of the generator even though it is not a usable feature.

5. **Arm sweep — pre-registered bars (written before the numbers; they do not move).**

   *Gate* (`s1` vs `s0`, 3-seed paired mean): ≥+0.0010 → full sweep; +0.0003…+0.0010 → cut
   sweep; <+0.0003 → `s4` only.

   **Gate result: Δ = +0.00055 ± 0.00011 → the "cut sweep" branch.** I nonetheless ran the full
   sweep (`s2`–`s5`) plus an extra arm `s6`. Recorded plainly because it is a deviation: the
   *bar* did not move and no verdict changed, but the gate's prescribed **action** was to run
   fewer arms, and I ran more. Justification — `s3` was an explicit user request, `s2` is the
   control without which no imputation delta is interpretable, and the box had idle capacity.
   Over-delivery on coverage, not a moved goalpost.

   *Adoption* (best arm vs `s0`): ≥+0.0010 → adopt. +0.0003…+0.0010 → **hold, and re-screen with
   `screening.screen_mechanism` before adopting**. <+0.0003 → flat.

   The band applies to the **plain paired mean**, not mean-minus-sd. Note the seed-to-seed sd of
   OOF AUC here is only ~0.0001 (OOF pools all 691k rows, unlike the 0.0005 *fold* spread), so
   deltas well below +0.0010 are easily *detectable*. That is exactly why the bar must not move:
   +0.0010 is a **deployability** threshold, not a detectability one, and root CLAUDE.md lesson
   #1 is that a variance reducer is significantly positive at every seed by construction.

   **Pre-registered expectation for the M-slope screen on `s1`** (recorded before running
   `screen_surface.py`): **b ≈ 0, i.e. information.** Mechanism: axis-aligned splits cannot form
   linear combinations, so handing the tree `other_screen_hours = daily − social − gaming − work`
   and `sgw_lb` directly is a re-expression it cannot synthesise — and `sgw_lb` is genuinely new
   information on the 13.9% of rows where `daily` is NaN. A re-expression that helps every member
   equally should *survive* averaging rather than shrink. If instead b ≈ −1, that prediction was
   wrong and `s1` is a variance reducer that must not be adopted.

6. Standard model breadth once the surface is settled: XGBoost, CatBoost, an MLP leg
   (`train_mlp.py`, needs an imputing surface), then Nelder-Mead blending on prob/logit/rank.

7. **Sweep CUT after `s2`/`s3` (2026-08-04 eve) — returning to the gate's prescribed action.**

   The `s1` gate landed at +0.00055, whose pre-registered action was **"cut sweep"**. The
   afternoon session ran the full sweep anyway and logged that honestly as a deviation. The
   remaining arms (`s4`, `s5`, `s6` at 6 seeds each, plus `s0` seeds 48–53 for the null band)
   are hereby **not run**. This is not a moved bar — it is finally obeying one.

   Three independent reasons converge:
   - `feedback_lb_blind_protocol`: discard candidates with expected effect <0.001. `s2` (the mask
     control) measured **+0.00050 vs `s0`** — i.e. it reproduces `s1` and adds nothing of its
     own: paired against **`s1`** over the two shared seeds it is **−0.0000355 / +0.0000355,
     mean 0.0000000**. Masks are worth exactly nothing on top of the budget features, which is
     what the marginal AUC ≈ 0.500 finding already predicted. The imputation arms are
     pre-registered in this very file as *expected flat for GBDTs*.
   - The 12-member null band's only consumer was screening the imputation arms. If those are
     flat at +0.00001 there is nothing to screen, so the 6 calibration seeds buy nothing.
   - For GBDT purposes **the surface is settled at `s1`**. The one open surface question — which
     imputing surface the MLP eats — is answered by the MLP's own CV, not by an LGBM arm delta.

   What this does *not* close: `s4`/`s5` as **MLP substrates**. They are still worth building if
   the MLP-on-`s3` leg proves it can reach the blend; the veto is on scoring them as LGBM arms.

8. **The two genuinely untouched levers, both with a better prior than any arm delta.**
   Day 4 of 27 with a single model family is where breadth pays.
   - **`tune_lgbm.py` has never run.** `LGBM_PARAMS` are hand-set (lr 0.05 / 127 leaves / 1000
     trees, no search). Tune on **`s1`**, not the `--arm s0` default.
   - **There is no second model family.** `train_xgboost.py`, `train_catboost.py`, `legs.py` and
     `ensemble.py` are all written and all unrun.

9. **PRE-REGISTRATION — Day 6 (2026-08-06): breadth + tuning.** Written before the first run of
   the day; the bars do not move.

   **The null for breadth is the LGBM seed-bag, not a single LGBM.** Adding XGB/Cat/MLP to a
   blend buys two different things: plain variance reduction from averaging more members, and
   genuinely different inductive bias. Seed-bagging the six existing `s1` seeds already buys the
   first, and costs nothing. So breadth is priced as

   > `blend([bag(lgbm×6), xgb, cat, mlp])` **vs** `bag(lgbm×6)`

   Pricing it against a single seed would credit breadth with variance reduction that is free —
   exactly the failure root CLAUDE.md lesson #1 names.

   | Arm | Instrument | Bar | Action |
   |---|---|---|---|
   | **Breadth** | blend OOF AUC − `bag(lgbm×6)` OOF AUC | ≥ +0.0010 | adopt as the submission leg |
   | | | +0.0003 … +0.0010 | hold; re-screen with `screening.screen_mechanism` first |
   | | | < +0.0003 | flat — report, do not adopt |
   | **Tuning** | 3-seed paired 5-fold `s1`, tuned − untuned | ≥ +0.0005 | adopt tuned params for the leg |
   | | | < +0.0005 | keep untuned; log the search range as covered |

   Tuning gets the lower bar because it *replaces* the leg rather than adding a member — there is
   no combiner above it to absorb the gain (lesson #7 cuts the other way here).

   **Honest expected outcome, stated so a clean negative counts as a result:**
   - XGB and CatBoost see the *same 27 features*; expect rank correlation ≳0.99 with LGBM, solo
     OOF within ±0.0005 of 0.96373, and a blend gain of **+0.0003…+0.0010** — i.e. most likely
     the *hold* band, not the adopt band.
   - The MLP is the only genuinely different inductive bias. Expect solo OOF **0.955–0.962**
     (below every GBDT); that is not a failure, its value is decorrelation. It is the member most
     likely to carry the blend past +0.0010.
   - Tuning: mild positive prior, **+0.0005…+0.0015**. `num_leaves=127 / lr=0.05` with early
     stopping is already a sane point, so a large gain would be surprising.

   If breadth lands under +0.0003 that closes *this blend*, not the avenue (lesson #13). The
   classes still untested afterwards: feature interactions, a different loss, a different target
   parameterisation.

   **Correction to the record**: steps 7–8 above describe the `s3` arm as run. It was not —
   `results/` holds no `s3` array and `arm_scores.json` has no `s3` key, so `impute.py` has never
   executed end to end. Treated as unrun and smoke-tested before the MLP depends on it.

10. **PRE-REGISTRATION — Day 10 (2026-08-10): quantile-bin target encoding of crosses.**
    Written before the screen runs; the bar does not move. Raised by the user, who correctly
    noted that "the three categoricals carry no marginal signal" is *not* an argument against
    TE of quantile-binned **continuous** features — a different mechanism entirely.

    **What is already dead, and why — single-feature quantile TE.** `E[y | qbin_q(x)]` is
    exactly the function a depth-1 tree on `x` outputs, and the boosters fit hundreds of those.
    It is a strict *coarsening* of information the model already has (it discards within-bin
    ordering). `social_share` alone, `gaming_share` alone, any single column alone: no route to
    a gain, and none will be built.

    **What is untested — crosses.** A k-way cell mean needs tree depth ≥ k *and* the interaction
    to survive regularisation. That is the one thing a GBDT genuinely under-fits, and step 4
    already established that the 0.9277 → 0.96312 gap "is genuine interaction structure".
    Candidate cells (all fold-safe via `kaggle.encoding.add_quantile_bin_features` →
    `add_fold_safe_target_encoding`, edges and means fit on the fold train split only):

    | cross | mechanism it targets |
    |---|---|
    | `social_share × daily` | fixed social hours mean different things at different totals |
    | `n_obs_sgw × sgw_lb` | how *tight* the bound is × where it sits — missingness × value |
    | `screen_block_nulls × sgw_lb` | the mask × value interaction step 3 left explicitly open |
    | `screen_share_of_awake × sleep_hours` | screen time displacing sleep |
    | `gaming_share × social_share` | composition mix at a fixed total |
    | `stress_ord × daily` | the only ordered categorical |

    **Instrument — held-out residual correction, not a retrained arm.** Building six TE arms at
    5-fold each costs hours; the question "is there uncaptured structure in these cells at all?"
    is answerable in seconds and strictly precedes it. Split train rows by `id % 2`. On half A
    estimate the count-shrunk mean residual `E[y − oof | cell]`; add it to `oof` on half B and
    measure the AUC change on B. Symmetrise (fit B, score A) and report the mean. `oof` is
    `bag(xgb×6)`, the current best leg. The cell means see A's labels only, so B is honest.

    **Bar: ≥ +0.0010 mean held-out delta → build the real TE arm. < +0.0010 → do not build.**
    Justified as follows: a deployed TE arm must clear the standing +0.0010 deployability
    threshold, and this screen fits the correction *directly on labels* with no model having to
    learn it — the most favourable possible reading of the same structure. A cross that cannot
    clear the bar under direct label fitting will not clear it after a retrain.

    **Instrument calibration (lesson #2) — both controls run on our own data, not imported.**
    - *Positive control*: cells = the exact-integer-`daily` indicator found 08-10, which has a
      measured OOF residual of −0.07…−0.20. The instrument must register this as positive, or it
      cannot detect real residual structure.
    - *Negative control*: random cell assignment with the cell-size distribution matched to each
      candidate. Must read ≈0. Anything a candidate scores inside the negative band is noise.

    **Honest expected outcome:** ≈0 for the pure-value crosses — at 691k rows and depth 6+, two-way
    interactions among present values are cheap for a booster to find, and step 4's interaction
    gap is most likely already being harvested. The missingness × value crosses
    (`n_obs_sgw × sgw_lb`, `screen_block_nulls × sgw_lb`) are the only ones with a real prior,
    because that class is flagged untested in step 3 and the integer-`daily` probe showed the
    model does leave residual structure on tree-hostile cells. **Prior: ~20% that any candidate
    clears +0.0010; most likely outcome is a clean negative.** A negative closes *these six
    crosses under this instrument*, not TE as a class (lesson #13) — higher-order crosses and
    crosses built on the residual itself would remain untested.

### `ensemble.py` — a conflation that would have stranded the neural leg

Found and fixed 2026-08-04 eve, before the MLP was launched rather than after. `ensemble.py`
required **every** leg to share one surface, so an MLP-on-`s3` blended with LGBM-on-`s1` would
have raised. That would have made arms `s3`/`s4` unusable — while the stated justification for
building them was precisely "an MLP cannot consume NaN".

Two different invariants were being run together:

| Operation | Actually requires |
|---|---|
| paired **arm-delta** read (`s1 − s0`) | the same surface — lesson #9: an altered train matrix reads systematically lower |
| **blending** two model families | the same `y`, the same row order, comparable folds — *not* the same feature matrix |

Cross-surface blending is now permitted via an explicit `--cross-surface s3,s4` and prints a
loud banner; discovery still never crosses a surface on its own, because a surface mix has
length `len(y)` either way and no downstream shape check can catch it.

---

## Experiments log

Surfaces are named `s0`…`s5` and are defined in `src/run_arm.py::ARMS`. **Only same-surface
arrays may be blended or stacked.** All arm runs are CPU, `n_jobs=12`, 5-fold, seeds 42/43/44.

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-08-04 | — | Scaffolded competition; data downloaded; EDA table above | — |
| 2026-08-04 | `lgbm_v1` | Baseline LGBM (CUDA), 21 features, no imputation, no mask features. Submission `lgbm_v1.csv` written + verified against sample_submission (296,302 rows, ids identical, ascending) | **0.96312** |
| 2026-08-04 | `bench_device` | One fold, cuda 53.8 s / AUC 0.96247 / best_iter 903 vs cpu 22.2 s / 0.96217 / 608. **Sweep locked to CPU.** | — |
| 2026-08-04 | `mask_mechanism` | Pre-registered mechanism test. MAR-AUC gap ≥0.02 on **1** column (bar ≥2); mask-matrix adversarial AUC **0.5649** (bar ≥0.60); 380 unseen-pattern test rows (0.128%). **FAIL → mechanism is the same.** Prior was null; null held. | — |
| 2026-08-04 | `probe_latent_index` | Best 1-D index `2·social + other + weekend` = **0.9238** on complete rows, below the 0.9277 fitted-linear ceiling and the 0.95 bar. **DO NOT ADOPT.** Budget substitution confirmed as an exact identity. | — |
| 2026-08-04 | `s0` | v1 surface re-measured on CPU for a same-device paired baseline. Seeds 42/43/44 = 0.96316 / 0.96330 / 0.96310 | **0.96319** |
| 2026-08-04 | `s1` | **GATE arm** — budget-constraint features, no imputation (27 features). 6 seeds; paired Δ vs `s0` = **+0.000548 ± 0.000086** (sd), positive at every seed | **0.963713** |
| 2026-08-04 | `screen_surface s1 vs s0` | M-slope, 6 members. **Decay table below.** Exponent **−0.063 ± 0.001**. Null-band calibration **FAILED** (null exponent +1.056 ± 0.750 → band (−0.443, +2.555), too wide to use) | — |
| 2026-08-06 | `bag_s1x6` | The **null leg** for the Day-6 breadth pre-registration: the six `s1` seeds averaged into one leg. `submissions/bag_s1x6.csv` + `test_bag_s1x6.npy` written; **no OOF array saved** (`ensemble.py`'s single-leg branch writes only the test side — the OOF is recomputed on demand from the six `oof_s1_s4*.npy`). Not a candidate; it is the thing breadth must beat. | — |
| 2026-08-06 | `cat_s1` seed 42 | First non-LGBM family. **Misses its own pre-registered prior (0.96373 ± 0.0005) by ~8×.** Unexplained on the day — the session died mid-run 87 s later. Diagnosed 08-07, see below. | **0.95953** |
| 2026-08-06 | — | **⚡ HARD POWER CUT at 09:40:53**, mid-run. `journalctl -b -2` holds no shutdown records at all (last line: routine `smartd`, 09:25:37) while files were still being written until 09:40:53. XGBoost, the MLP and the tuning run produced nothing; work sat uncommitted for 30 h. Recovered 08-07 in `d4171aa`. | — |

### Day 7 (2026-08-07)

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-08-07 | `tune_lgbm` hardening | `create_study` had no `storage`, so a search that runs for hours died whole on any crash. Now sqlite-backed + `load_if_exists`, so the identical command resumes. **Verified against a real restart**: "resuming study 'lgbm_s1' … 2 completed trials". A crash now costs one trial. | — |
| 2026-08-07 | `smoke_impute` | `impute.py` had never executed end to end. Found `median_impute` (`s3`) violating the budget identity by **−5.23 h** — independent per-column medians let imputed parts exceed an *observed* `daily`, so `_recompute_derived` rebuilt `other_screen_hours` **negative**. `s3` is only in the sweep as the honest control for `s4`, so a broken `s3` would have let `s4` win partly by not being broken. Fixed with `s4`'s phase structure, naive median estimator kept. All four arms now PASS at `min(daily − sgw_lb) = −0.000000`. | — |

### Day 10 (2026-08-10) — the Day-6 breadth pre-registration, executed and closed

Bars quoted from item 9 and **not moved**. The null is `bag(lgbm×6)` = **0.964346**, so the bands
in absolute terms are: flat < 0.964646 | hold 0.964646–0.965346 | adopt ≥ 0.965346.

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-08-10 | `diag_catboost` | **H1 CONFIRMED.** CatBoost defaults `border_count` to 254 on CPU but **128 on GPU**; the shipped recipe took the GPU default. Fold 1: A border=128 **0.95863** (105.4 s) vs B border=254 **0.96148** (82.3 s) — recovers **+0.00285 of the −0.00422 gap (68%)** *and is faster*. H2 (early stopping) untested: configs C/D were cut, see deviations. | — |
| 2026-08-10 | `cat_s1` ×3 (fixed) | `border_count=254` patched into `train_catboost.py`. Seeds 42/43/44 = 0.96225 / 0.96226 / 0.96226 — the 5-fold recovery **+0.00272** matches the fold-1 prediction (+0.00285) almost exactly. Bagged `cat_s1x3` = **0.962516**, rank corr 0.9880 vs the null. | **0.96226** |
| 2026-08-10 | `xgb_s1` ×6 | First XGBoost leg. Seeds 42–47 land in **0.96449–0.96454** (sd ≈0.000015 — far tighter than LightGBM's 0.000043). **A single XGB seed beats the entire 6-seed LightGBM bag.** Misses its own pre-registered prior (0.96373 ± 0.0005) on the *high* side. | **0.96452** (s42) |
| 2026-08-10 | `mlp_s4` seed 42 | The pre-registration named this "the member most likely to carry the blend past +0.0010". It came in at **0.94198**, −0.013 below its 0.955–0.962 prior, and per-fold 0.9410–0.9429 — stably weak, not unstable. **Harmful at every weight** against both the LGBM bag alone and the bag+xgb blend (monotone decreasing from w=0). Seeds 43/44 cut. | **0.94198** |
| 2026-08-10 | **breadth blend** | Nelder-Mead over `bag(lgbm×6)` 0.964346, `bag(xgb×6)` **0.965066**, `bag(cat×3)` 0.962600, `mlp_s4` 0.941983. All three spaces tie at 0.96509; weights `xgb 0.851 / lgbm 0.149 / cat 0.0 / mlp 0.0`. LR-on-logits stack **0.96479 — worse than the blend and worse than xgb alone**. `submissions/blend_breadth.csv`, all verify checks PASS. | **0.965090** |
| 2026-08-10 | `screen_breadth` | Hold-band action. Fixed-weight M-slope, M=1..4: +0.000926 / +0.000825 / +0.000777 / +0.000731, **b = −0.167**. At M=4 the observed delta is **3.2×** what 1/M variance reduction predicts and **1.6×** what M^−0.5 selection noise predicts; retention 79%. **Mechanism = information.** | — |
| 2026-08-10 | `probe` — generative-rule hypothesis (forum) | Applied the "hard rule" `D>8 or S>4 → 1`, `D≤6 and S≤4 → 0` to all 691k train rows. The **"certain-0" region is 32.5% positive** and the "pure-noise" band scores AUC 0.924 under our model. Rule-as-predictor **0.8322**; forcing both deterministic regions to be called perfectly **lowers** AUC to 0.9245. A 0.1-wide scan of `D` 5.5→8.5 at `S≤4` is a monotone ramp with no step at 6 or 8. **Independently re-confirms step 4** (noisy sigmoid of a latent index, not a hard rule). No lever. | — |
| 2026-08-10 | `probe` — exact-value artifact | New finding, not previously in the log: exact-integer `daily_screen_time_hours` rows run *below* their immediate neighbours — `D=7.00` y=0.192 vs 0.639 at `D∈(7.00,7.03]`, survives conditioning on `S≤4`, OOF residual −0.17 (~21σ). Trees cannot build an exact-value indicator, so it is the right *class*. But it is 0.54% of rows: oracle pricing gives **−0.000121** (group rate) and **+0.000392** (those rows labelled *perfectly*). Widening to all round-valued rows in 6 columns (4.37%) with perfect labels: +0.003320. **Closed — the attainable version is far under the +0.0010 bar.** | — |
| 2026-08-10 | **arms `s7`/`s8` — TE, direct read** | The deliverable, not a proxy. XGB seed 42, CPU, same folds/params as `xgb_s1` s42 = **0.964524**. `s7` (`s1` + TE of the 6 crosses, 33 feats) = **0.964371, Δ −0.000152**. `s8` (`s1` + TE of 8 single q=64 bins, 35 feats) = **0.964398, Δ −0.000125**. **Both lose on all 5 of 5 folds** — a consistent paired negative, not seed noise. TE **rejected on the direct read**, not merely on the screen. Verified not a no-op: OOF differs from `xgb_s1` (corr 0.9971, max\|diff\| 0.40). Method note: the residual screen below predicted −0.000057 and the direct read landed at −0.000152 — right sign, right order of magnitude, so the proxy was *directionally* sound, but treating it as decisive was wrong when the direct read costs 11 min/arm. **Lesson: price the experiment before substituting a proxy for it.** | −0.00015 |
| 2026-08-10 | `legs.py` reporting bug | `run_leg` printed `features={x.shape[1]}` from the **pre**-transform matrix, so every arm with a transform (`s2`–`s5`, `s7`, `s8`) reported the bare `s1` count of 27 and was indistinguishable from a transform that silently no-op'd. Now reports post-transform width and **raises** if a transform adds no columns. Found because `s7` printed 27 where 33 was expected. | — |
| 2026-08-10 | `screen_te_crosses` (item 10) | **Screen only — superseded by the direct read above.** Verdict agreed, but see that row: the screen measures an *additive* per-cell correction and structurally cannot see a TE column being split on and interacted with the other 27 features, so it was never entitled to close the question alone. **CLEAN NEGATIVE against the unmoved +0.0010 bar.** All six pre-registered crosses returned a *negative* held-out residual-correction delta; best was `n_obs_sgw × sgw_lb` at **−0.000057**, worst `gaming_share × social_share` at −0.000343. **First positive control was mis-designed** (integer-`daily` cells — structure already proven worthless in AUC terms, so its null reading was uninformative); replaced with a corruption-recovery control, which recovers **78% / 82% / 90%** of injected per-cell noise at sd 0.02/0.05/0.10 and registers **+0.001145** against damage of only −0.001471 — sensitivity at the bar's own scale. Negative-control band −0.000368…−0.000124; 5 of 6 candidates sit *above* their matched control, so a trace of real cell structure exists at ~+0.0002 — an order of magnitude under bar. Closes **these six crosses under this instrument**, not TE as a class. | — |

**Verdict against the unmoved bar.**

| Quantity | Value |
|---|---|
| null `bag(lgbm×6)` | 0.964346 |
| `bag(xgb×6)` | 0.965066 (**+0.000720** vs null) |
| best blend (logit) | 0.965090 (**+0.000744** vs null) ← the pre-registered instrument |
| blend − `bag(xgb×6)` | **+0.000024** ← what *combining* actually contributes |

**HOLD** (+0.0003…+0.0010). The adopt threshold is +0.0010 and it was not reached. The delta also
*passes* the mechanism screen (information, not variance reduction) — two separate facts, and
neither promotes the other. No promotion rule was invented after the number landed.

**The headline is not "breadth pays" — it is "XGBoost is the better model here".** 97% of the
+0.000744 is `bag(xgb×6)` simply beating `bag(lgbm×6)`; the act of combining them adds
**+0.000024**, which is a quarter of the +0.0001 OOF resolution and indistinguishable from zero.
A four-family blend collapses to one family plus a rounding error. The LR stack is *worse* than
its own best leg (0.96479 vs 0.965066), which is the same message stated more bluntly.

Practical consequence: `bag(xgb×6)` alone is a legitimate submission — it is 0.000024 behind the
blend, has no fitted weights to overfit, and is far simpler. Prefer it unless a later leg makes
combining actually pay.

What this closes is *this blend*; the classes still untested are feature interactions, a different
loss, a different target parameterisation, and a *better* neural leg (not merely more seeds of
this one). **Tuning XGBoost is the armed next lever** — `src/tune_xgboost.py`, written 08-10 —
because tuning replaces the leg rather than adding a member, so lesson #7's combiner absorption
does not apply to it.

**Three of four pre-registered expectations were wrong, and the misses are the informative part:**

- **XGB was predicted redundant** (rank corr ≳0.99, solo within ±0.0005). Corr was 0.9952 as
  predicted, but it is the *strongest* leg and takes ~0.78 of the blend weight. Correlation
  predicted redundancy; it did not deliver it.
- **The MLP was predicted to be the decisive member.** It is the most decorrelated leg by a wide
  margin (0.9713) and is still worthless — decorrelation is necessary, not sufficient. A leg
  0.022 below the null cannot buy its way in on diversity.
- **CatBoost's gap was predicted to be a recipe artifact.** It was, exactly. And repairing it moved
  the deliverable by **0.000000** — root CLAUDE.md lesson #7 (a strong combiner absorbs leg-level
  gains) demonstrated on a leg whose repair was worth +0.00272 solo.

**Deviations from the day's plan, recorded plainly:**

1. **Optuna killed** ~17 min in, having completed **0 trials** while starving the breadth runs
   (load average 32 on 12 cores; `nice -n 19` does not help when the contention is memory
   bandwidth). Sqlite-resumable, so nothing was lost. The study still stands at 6 trials with
   **trial 0 — the enqueued untuned reference, 0.96336 — as the best**; trials 1–5 all lost to it.
2. **`diag_catboost` cut after config A**, then re-run for config B only with `--reference`
   supplied rather than refit. H2 (early stopping / iteration count) is therefore **untested**.
   Justification: the diagnosis only mattered if CatBoost could become non-harmful, and once the
   fixed leg still scored weight 0.0 the remaining configs could not change a decision.
3. **MLP cut to 1 seed** rather than 3, once it measured harmful at every weight.
4. **The GPU collision was self-inflicted**: starting the MLP while `diag_catboost` held
   `gpu_ram_part=0.8` left it 142 MiB and OOM-killed it. `gpu_ram_part` is now env-overridable.

**Three bugs fixed today, all pre-existing:**

- **`ensemble.py` wrote a constant-zero submission.** `clipped_logit` is **misnamed** — it returns
  `log(p)`, not `log(p/(1−p))` (its own docstring says so). So the winning `logit` blend came out
  all-negative and `np.clip(., 0, 1)` collapsed every one of the 296,302 rows to 0.0. Caught by
  `verify_submission.py`; it had never fired before because **this was the first multi-leg blend
  the competition ever ran** (the single-leg `bag_s1x6` branch writes `tests[0]` untransformed and
  is unaffected). Fixed with an explicit `INVERSE` map per space plus a constant-output guard that
  raises rather than writes. `clipped_logit` itself was **not** changed — s6e6/s6e7 stacking
  depends on its current behaviour, and renaming it is a separate job.

- `data_science_stuff.kaggle.models.mlp.fit_mlp_fold` forwarded the **entire** val and test
  splits through the net in one pass (~600 MB of activation per layer at 296k rows × width 512),
  which OOMs a 6 GB card. Now batched via `_predict_batched`; peak inference memory is independent
  of split size. This affects **every** competition using this scaffold, not just s6e8.
- `legs.run_leg` printed nothing until all five folds finished, so a silent log was
  indistinguishable from a hung or dead run — which is how the 08-06/08-07 sessions lost work.
  Now reports per fold via `run_cv`'s `after_fold` hook.

### PRE-REGISTRATION — XGBoost tuning (written 2026-08-10 16:25, before any trial but the reference)

The Day-6 tuning bar was written for LightGBM. It is hereby applied **unchanged** to XGBoost,
which is the leg that now carries ~0.85 of the blend weight. Recorded before the numbers exist.

| Instrument | Bar | Action |
|---|---|---|
| 3-seed paired **5-fold** `s1`, `xgbt` − `xgb` | ≥ +0.0005 | adopt tuned params for the leg |
| | < +0.0005 | keep untuned; log the search range as covered |

**The Optuna number is for ranking only and is NOT the instrument.** The search runs 3 folds
(reference trial 0 = **0.96409**, vs the same params' 5-fold OOF of 0.96452 — 3 folds read ~0.0004
lower because each model sees less data). Adoption is decided on a 5-fold paired refit, which is
why `train_xgboost.py --tuned` writes `xgbt_` arrays rather than overwriting `xgb_`: the paired
read needs both arms to coexist.

**Honest expected outcome:** mild positive, **+0.0003…+0.0010**. `lr 0.03 / depth 8 / 3000 trees`
with early stopping is already a sane point, and the sibling LightGBM study is the cautionary
case — after 6 trials its enqueued untuned reference is *still* the best, so a well-chosen hand
setting is a real possibility here too. A clean negative against this bar is a result, and the
falsifier is explicit: if the tuned 5-fold paired delta lands under +0.0005, the hand-set params
stay and the range is logged as covered.

Tuning is the one lever lesson #7 does not blunt: it **replaces** the leg rather than adding a
member, so there is no combiner above it to absorb the gain — unlike today's CatBoost repair,
which was worth +0.00272 solo and +0.000000 to the blend.

### PRE-REGISTRATION — Day 11 (2026-08-11): pairwise objective + RealMLP

Written 09:30, **before any run of the day**, including before the RealMLP fold-1 smoke. The
bars below do not move. The Day-10 XGBoost tuning bar above is carried forward **unchanged** and
is executed today, not re-derived.

**Framing that sets both bars.** Day 10 measured combining four families at **+0.000024** over
`bag(xgb×6)` alone. So in this competition an *added member* has almost no route to the
deliverable, while a *leg replacement* does (lesson #7). That is why the pairwise objective is
priced on both routes, and why the neural leg is priced at the blend rather than solo.

| Arm | Instrument | Bar | Action |
|---|---|---|---|
| `xgbp` **as replacement** | 3-seed paired 5-fold `s1`, `xgbp` − `xgb` solo OOF | ≥ +0.0005 | adopt as the leg |
| `xgbp` **as member** | `blend([bag(xgb×6), xgbp]) − bag(xgb×6)` | ≥ +0.0005 | adopt as a member |
| `realmlp` | `blend([bag(lgbm×6), bag(xgb×6), realmlp]) − blend([bag(lgbm×6), bag(xgb×6)])` | ≥ +0.0005 | adopt as a member |

**Honest expected outcomes, so a clean negative counts as a result:**

- **`xgbp` as a replacement: flat to slightly negative.** `binary:logistic` is a proper scoring
  rule and already a strong AUC surrogate at 691k rows. Pairwise objectives win where the metric
  is *top-heavy* (NDCG@k), and AUC weights every pair equally — which is also why
  `lambdarank_pair_method` must be `"mean"`, not the `"topk"` default.
- **`xgbp` as a member: the better of its two priors**, because a pairwise-trained ensemble
  should make different errors in the tails. **~25% that either route clears.**
- **`realmlp`: solo 0.950–0.960, blend delta +0.0000…+0.0002 — most likely weight 0.0.** The
  governing datum is not `mlp_s4`'s 0.942 but `bag(cat×3)`'s **0.962516**, which is a mere 0.0018
  below the null and *still* earned weight exactly 0.0. Decorrelation is not the argument: the
  MLP was the most decorrelated leg (0.9713) and was harmful at every weight.

**Stop-loss for the neural track (a routing rule, not a bar):** if `realmlp` seed 42 lands under
**0.955** solo OOF, seeds 43/44 are not run. A leg 0.010 below the null cannot buy in when one
0.0018 below it could not.

**Ideas priced and dropped before spending anything on them** (user-raised, judged 09:30):

- **Yeo-Johnson / power transform on inputs — dropped, no measurement taken.** It is already
  applied where it can matter: `fit_mlp_fold` fits `PowerTransformer(yeo-johnson, standardize=True)`
  on the fold-train split. And it is *inert* where it isn't: for any strictly monotone φ, the split
  `φ(x) < φ(t)` selects the identical row set, so no per-feature monotone transform can change an
  LGBM/XGB tree. Not a low prior — a no-op.
- **Power transform on the output — dropped, structurally impossible.** The target is binary 0/1
  (there is no continuous output), and ROC AUC is invariant under any monotone transform of the
  predictions. This file already contains the demonstration: the prob/logit/rank blends **tied**
  at 0.96509 on Day 10.
- The live reading of "a different target parameterisation" is therefore not a transform of the
  output at all — it is a different training *objective*, which is `xgbp`.

### Day 13 (2026-08-13) — the XGBoost tuning bar, executed and CLEARED

**Two days were lost to a hard reboot.** The 08-11 session died at ~11:00 mid-`xgbt` (fold 4/5 of
seed 42, `logs/xgbt_s1.log`), and the box was not brought back up until 08-13 09:19. This is the
second unplanned power event of the campaign (cf. the 08-06 cut) and the same box the
`~/sysmon` flight recorder was installed for. Nothing was lost this time beyond wall clock —
the Day-11 pre-registration above was already written to disk before the run started, which is
exactly what made today's read honest rather than post-hoc. It was, however, sitting
**uncommitted for two days**; committed first thing today before any new run.

**The Day-10 tuning bar is quoted from above and was not moved:** 3-seed paired 5-fold `s1`,
`xgbt` − `xgb`, **≥ +0.0005 → adopt**.

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-08-13 | `xgbt_s1` ×3 | The pre-registered read, seeds 42/43/44 = 0.965668 / 0.965703 / 0.965622 against the untuned 0.964524 / 0.964545 / 0.964506. Paired Δ = **+0.001144 / +0.001158 / +0.001116**, mean **+0.001139 ± 0.000021**, positive on every seed. **≥ +0.0005 → ADOPT the tuned params for the leg.** | **0.965664** (3-seed mean) |

**This is the largest single lever the campaign has found, and it beats the whole Day-10 blend.**

| Quantity | Value | vs null `bag(lgbm×6)` 0.964346 |
|---|---|---|
| `bag(xgb×6)` (untuned) | 0.965066 | +0.000720 |
| Day-10 four-family blend | 0.965090 | +0.000744 |
| **one tuned seed `xgbt_s1_s42`** | **0.965668** | **+0.001322** |

A *single tuned XGBoost seed* is **+0.000578** ahead of a four-family Nelder-Mead blend of
sixteen models. That is the cleanest possible statement of root CLAUDE.md lesson #7: the
combiner absorbed everything breadth had to offer (+0.000024 for the act of combining), and a
lever that **replaces** the leg walked straight past it.

**The pre-registered expectation was too pessimistic, and the reason is on the record.** The
prior said "+0.0003…+0.0010, mild positive" and cited the sibling LightGBM study — where the
enqueued untuned reference *still* beats 11 tuned trials — as the cautionary case. The XGB search
landed at 2.3× the top of that range. The two studies differ in one identified way: the LightGBM
space was never recentred after its trials lost, while the XGB space was recentred once (`v2`,
commit `9a67ae3`) and again at `v3`, and only the `v3` study found anything (trial 8, 3-fold
0.96530 vs the reference's 0.96409). **A search space that never beats its own enqueued reference
is a mis-centred space, not evidence that tuning is dead** — the LightGBM study should be read
that way rather than as a null.

**Method note kept for the next competition:** the 3-fold Optuna number (0.96530) and the 5-fold
paired read (0.965664) differ by +0.00036, in the direction and roughly the magnitude the
pre-registration predicted ("3 folds read ~0.0004 lower because each model sees less data"). The
ranking proxy was sound *and* was not treated as the instrument — the adopt decision was taken on
the 5-fold refit, as written.

### Day 15 (2026-08-15) — backfill, the bag, and the capacity pre-registration

**Backfill of what happened after commit `4fcd464` and was never written down.** The 08-13
session continued past the commit and then stopped; 08-14 was a no-work day.

| Date | Run | Description | OOF AUC |
|---|---|---|---|
| 2026-08-13 | `xgbt_s1` seeds 45/46/47 | Ran 16:50–17:13, never logged. 0.9656252 / 0.9656604 / 0.9656322. With 42/43/44 that is a full six-seed set; 6-seed mean **0.9656518**, sd ≈ 0.00003 (LGBM's is 0.000043). | 0.9656518 |
| 2026-08-13 | `tune_xgboost` v3 resume | Trials 11/12/13 = 0.96375 / 0.96506 / 0.96475 at 3 folds. **None beat trial 8's 0.96530.** Study now 14 of its 60-trial budget; trial 14 was started 17:41:36 and abandoned mid-flight (stale RUNNING row). `best_params_xgboost_v3b.json` is byte-identical to the adopted file but for `n_jobs`, because nothing beat the incumbent. | — |
| 2026-08-13 | `realmlp_s4` s42 | **Interrupted, incomplete.** Fold 1 = 0.95621 (24-epoch run, peak at epoch 10, then monotone decay to 0.95000 — clean overfit, so `epoch_select="auc"` is load-bearing). The 5-fold run got through fold 1 in 616 s and died; `results/realmlp_ckpt/s42_f{1,2}.pth` exist, no OOF array. Against the Day-11 stop-loss (<0.955 → cut seeds 43/44) the fold-1 read **clears**, so the arm is live but unclosed. | — |
| 2026-08-15 | **`bag(xgbt×6)`** | The tuned leg had never been bagged or submitted. Six seeds averaged → **0.96609**, `submissions/bag_xgbt_s1x6.csv`, all eight `verify_submission` checks PASS (296,302 rows, ids elementwise-identical, sd 0.368, mean 0.7096). **+0.001744 over the old null and +0.001 over the Day-10 blend.** | **0.96609** |

**The finding that motivates today's lever: the adopted config never early-stopped.**

Queried read-only from `results/optuna_xgb_s1.db`, study `xgb_s1_v3`, per-trial `best_iteration`:

| trial | 3-fold AUC | `best_iteration` |
|---|---|---|
| 0 (reference) | 0.96405 | 1929 |
| 7 | 0.96465 | 766 |
| **8 — adopted** | **0.96530** | **2999** |
| 10 | 0.96447 | 2999 |
| 6 | 0.96414 | 2998 |
| 12 | 0.96506 | 778 |

Trial 8 used **2999 of `n_estimators = 3000`**. Early stopping at patience 100 never fired — the
adopted point was **cut off, not converged**. Two more of its parameters were likewise selected at
the edge of the box they were searched in: `max_bin = 512` is the **top** of the searched set
`{256, 512}`, and the `learning_rate` floor of 0.015 was never breached by any trial. The search
found the best point it could *reach* and then hit the wall of its own bounds.

### PRE-REGISTRATION — Day 15 (2026-08-15): the capacity ceiling

Written **before `diag_capacity.py` was run** and before any number from it exists. The bar does
not move.

**Why this lever and not another.** Day 13 measured the combiner contributing **+0.000024** while
a leg *replacement* paid **+0.001139**. Raising a capacity ceiling replaces the leg, so lesson #7
cannot absorb it. This is the same class as the lever that just paid.

**One factor at a time.** `max_bin` 512→1024 is a *different* mechanism (binning resolution, with
independent corroboration: the CatBoost `border_count` 128→254 repair was worth +0.00285 solo, and
`max_bin=128` was dropped from the XGB space for landing among the worst trials). Running it in
the same arm as `n_estimators` would make a negative uninterpretable. Capacity leads; `max_bin` is
queued behind it as its own read.

**Stage 1 — diagnostic, mechanistic not score-based.** `src/diag_capacity.py`, fold 1 of the
`legs.py` 5-fold split, seed 42, arm `s1`. Config A = adopted params exactly. Config B =
`n_estimators` 3000 → 10000, nothing else changed. One fold cannot resolve +0.0005 (fold sd ≈
0.0005) and is **not** being asked to; it is asked only "is there headroom at all?", which is a
question about where ES fires and which one fold answers definitively.

**Gate: promote to stage 2 only if `best_iteration ≥ 3500`.** If it stops near 3100 the config was
converged all along, the lever is dead, and the cost was 15 minutes.

**Stage 2 — the read.**

| Instrument | Bar | Action |
|---|---|---|
| 3-seed paired 5-fold `s1`, `xgbc` − `xgbt` | ≥ +0.0005 | adopt the raised cap for the leg |
| | < +0.0005 | keep the cap at 3000; log the capacity axis as covered |

The bar is the **standing tuning bar carried forward unchanged** — same instrument and same class
of lever as the 08-13 read. It is not re-derived and it is not relaxed.

**Honest expected outcome: +0.0000…+0.0004, most likely UNDER the bar. ~30% it clears.**
*For*: ES genuinely never fired, and the best point the search found is the one that ran out of
rope. *Against*: patience-100 not firing means "no degradation for 100 rounds", not "still
climbing", and the XGB AUC tail at 691k rows and lr 0.029 is very flat. A clean negative against
the unmoved bar is a successful experiment, not a wasted afternoon.

**Confound named before the number, not after: this is the lever most exposed to stopping-rule
optimism.** This file already records that every leg early-stops on the fold it then writes as its
OOF, and that XGB's longer leash makes it the worse offender of the two. Extending that leash from
3000 to 10000 **amplifies exactly that bias** — more rounds is more opportunity to select a
stopping point that flatters the very fold being scored.

**Guard, pre-registered so it cannot be quietly dropped if the number is nice:** if `xgbc` clears
+0.0005 it is **not adopted on that read alone**. Re-verify with a fixed-round refit at the
discovered `N*` with **early stopping OFF**, so the val fold is never used for selection. If the
gain survives, it is capacity. If it evaporates, it was stopping-rule optimism and the cap stays
at 3000.

### M-slope screen: `s1` vs `s0` (6 members, 2026-08-04)

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| delta | +0.000549 | +0.000514 | +0.000502 | +0.000496 | +0.000492 | +0.000490 |
| se | ±0.000005 | ±0.000005 | ±0.000005 | ±0.000005 | ±0.000005 | ±0.000005 |

**The printed verdict was "indistinguishable from selection noise — dead". I am recording that
verdict as an artefact and adopting `s1` anyway. The reasoning, stated so it can be checked:**

The printed verdict compares the exponent against a `null_band` that this run **failed to
calibrate, for an identifiable and fixable reason**:

`selection_null_arms` builds its known-fake control by splitting the members into **two disjoint
halves**. With 6 members each null arm gets **3**, so `screen_mechanism`'s `size_max` collapses
to 3 regardless of the `max_size=6` that was passed — and the null exponent was therefore **fit
on three points**. That is precisely the failure root CLAUDE.md lesson #4 exists to name
("do not fit an exponent on three points"; s6e7 published +0.475 and corrected to −0.158 at
six). Hence sd 0.750 and a band three exponent-units wide that would swallow *any* result,
including a perfect b = −1.

**The band is unusable because the calibration was under-powered, not because the data is
uninformative and not because the layer broke the selection step** — the M=1 orientation does
happen correctly with the AUC layer. Fix: a 6-member null needs **12** stock members. Seeds
48–53 of `s0` are queued for exactly this, so the imputation/`s6` screens get a usable band.

The adoption decision therefore rests on the **directly observed retention**, which is a
measured quantity, not a fitted parameter, and needs no band:

| | M=6 delta | vs observed |
|---|---|---|
| **observed** | **+0.000490** (se ±0.000005) | — |
| if variance reduction (~1/M) | +0.000092 | observed is **5.3×** larger |
| if selection noise (~M^−0.5) | +0.000224 | observed is **2.2×** larger |

Retention at M=6 is **0.000490 / 0.000549 = 89%**. Both null mechanisms predict values far
outside the se on the observed delta, so both are falsified by direct observation.

Structurally, selection noise also cannot apply *to `s1` specifically*: the winner's curse
requires selecting the best of many candidates, and `s1` was pre-registered as the single gate
arm before any number existed. There was no selection to be cursed by. **This argument does not
transfer to `s6` or to the imputation arms** — those are chosen as the best of several, so they
need the properly-powered band.

**This overrides a tool verdict in the same direction as my own pre-registered prediction
(b ≈ 0), which is exactly the shape of motivated reasoning — so the falsifier is stated
explicitly: if the M=6 delta had come in at or below ~0.00022, I would have been wrong and `s1`
would be dead. It came in at 0.00049.**

Residual honesty: ~11% of the M=1 advantage *does* decay away, so a small variance-reduction
component is present. The other 89% is not.
