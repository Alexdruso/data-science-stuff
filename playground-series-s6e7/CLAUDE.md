# PS S6E7 — Predicting Student Health Risk

**Task**: 3-class classification — predict `health_condition` ∈ {`at-risk`, `fit`, `unhealthy`}
from lifestyle/biometric features.
**Metric**: Balanced accuracy (mean per-class recall).
**Deadline**: ~August 2, 2026.

---

## ⚠️ CRITICAL: Row-order invariant — always load labels and IDs through build_features()

`features.py::build_features()` sorts every dataframe by **`["id"]`** (`SORT_KEY`). All
`results/oof_<model>.npy` and `results/test_<model>.npy` arrays are stored in this sorted order.

Any script that loads `y` or `test_ids` and combines them with the npy arrays **must** go through
`build_features()`, or predictions silently misalign → garbage submission:

```python
# CORRECT
train_pl = build_features(pl.read_csv(DATA_DIR / "train.csv"))
y = train_pl[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — raw CSV order ≠ npy order
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

### Label ↔ index ↔ proba-column invariant

`features.CLASSES = ["at-risk", "fit", "unhealthy"]` is the fixed, alphabetical order. It equals
the integer encoding (0/1/2) AND every model's `predict_proba` column order (sklearn/LGBM/XGB
sort `classes_`; CatBoost with integer labels does too). `train_common.finalize()` asserts
`oof.shape == (690088, 3)` / `test.shape == (295753, 3)` and that rows sum to 1; each trainer
asserts `model.classes_ == [0, 1, 2]`. Decode with `np.array(CLASSES)[pred_idx]`.

---

## DRY conventions

- **`features.py`** is the single source of truth for transforms, `CLASSES`, `SORT_KEY`,
  `EXCLUDE_COLS`, `CAT_COLS`, `NUM_COLS`.
- **`train_common.py`** holds the shared plumbing: `load_dataset()` (row-order + label encoding
  + alignment) and `finalize()` (decision-weight tuning on OOF, `save_cv_result`, npy saves,
  per-model submission). Every `train_*.py` calls these — do not re-implement the CV loop's
  bookkeeping.
- **`cv_results.py`** is a thin re-export of `data_science_stuff.kaggle_utils.save_cv_result`.
- Decision-weight tuning / reporting reuse `kaggle_utils` (`tune_decision_weights`,
  `weighted_predict`, `classification_report_dict`) — the same helpers as S6E6.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 690,088 | 15 (13 features + id + target) |
| test  | 295,753 | 14 (13 features + id) |

**Class balance**: at-risk 85.9% · unhealthy 8.4% · fit 5.8% — heavily imbalanced (all-`at-risk`
scores only 0.333 balanced accuracy).
**Missing values**: in **every** feature column (heart_rate 1.1% … stress_level 12.0%). GBDTs
handle NaN natively; CatBoost categoricals are filled with a sentinel string.

### Raw features

- **Numeric (7)**: `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`,
  `exercise_duration`, `water_intake`.
- **Categorical (6)**: `diet_type` (veg/non-veg/balanced), `stress_level` (low/medium/high, ordinal),
  `sleep_quality` (poor/average/good, ordinal), `physical_activity_level`
  (sedentary/moderate/active, ordinal), `smoking_alcohol` (no/occasional/yes, ordinal),
  `gender` (male/female/other).

---

## Key EDA findings (`src/eda.py` → `results/eda_summary.md`)

- **`stress_level` is near-deterministic**: `high` → 85.8% of unhealthy; `low` → 84.5% of fit;
  `medium` → mostly at-risk. Single strongest signal.
- **`physical_activity_level=active`** → 91.7% of fit. `sleep_quality`, `smoking_alcohol` also
  strongly split fit vs unhealthy.
- **Numerics separate fit**: fit has higher sleep_duration (7.95 vs 5.37 unhealthy),
  step_count (11.7k vs 8.4k), exercise_duration (50 vs 38); bmi higher for unhealthy (24.1 vs 21.8).
- **`water_intake` is pure noise** (identical mean across all classes); `diet_type` and the
  `occasional` level of `smoking_alcohol` carry ~no signal.
- **Missingness ≈ MCAR** — null rates are ~uniform across classes, so missingness-indicator
  features are expected to be low-value.
- **Adversarial validation AUC = 0.65** (moderate train/test covariate shift), driven by the
  numeric features + a `gender` share shift (female 32.5%→29.2%, other 30.0%→33.1%).
  **BUT CV tracks LB almost exactly** (see below) — the shift did not break validation; trust OOF.

---

## Current best — STILL ensemble_v1b, LB 0.94970 (do not regress from it)

**Champion = `ensemble_v1b`: non-bagged 3-GBDT decision-corrected blend, water_intake KEPT →
LB 0.94970.** Everything tried on Day 2 (bagging, drop water_intake, add MLP) has **higher OOF
(0.9495) but LOWER LB (0.94910)** — a −0.0006 regression. Keep v1b as a final submission.

### ⚠️ Day-2 lesson: OOF is saturated — it cannot rank configs at this ceiling
OOF balanced-acc sits at 0.949x for every config; differences <0.001 are **shift-noise, not
signal** (adv-AUC 0.59–0.65). Concretely: ensemble v2 had the *highest* OOF yet the *lowest* LB.
`src/adv_eval.py` (adversarial-validation-weighted OOF) shows why the **MLP does not transfer**:
on the top-30% most test-like train rows, 3-GBDT and 3-GBDT+MLP are **tied (0.9483)** — the MLP's
OOF gain lives entirely in *non*-test-like rows. **Rule: stop optimizing OOF; only the LB (and, as
a proxy, the test-like-subset bacc from `adv_eval.py`) can rank near-ties. Change nothing off a
<0.001 OOF move.**

### Day-2 PM: offline instruments (no submissions left) — hard findings

- **`lb_anchor.py` — offline metrics FAILED validation as fine rankers.** Reconstructed all 5
  scored configs and ranked them with 8 candidate metrics against known LB: best Kendall tau 0.4
  (adv-weighted bacc); nothing reproduces the sub-0.0002 orderings. The test-like/adv-weighted
  family does fix the one costly inversion (puts v1b ≥ v2 where plain OOF said v2 > v1b), but by
  paper-thin margins. **Rule: offline metrics = coarse veto only; only the LB ranks <0.001 gaps;
  offline "wins" are actionable only at >0.001 margins.** Report: `results/lb_anchor_report.txt`.
- **`forensics_v1b_v2.py` — v2's LB damage is localized.** v1b vs v2 differ on only 1,023 test
  rows (0.35%); 79% are at-risk↔unhealthy swaps and **84% of flipped rows have a missing key
  driver** (base 26%). v2 re-drew the at-risk↔unhealthy boundary in the information-limited
  missing-driver region and lost. **Rule: don't move the missing-driver decision boundary without
  LB proof.** Champion reconstruction reproduces OOF 0.9494 exactly (pipeline reproducible;
  labels → `results/champion_test_labels.npy`).
- **Duplicate probe: DEAD.** Zero exact train↔test feature matches; zero within-train duplicates.
- Env-gated reproducibility added: `S6E7_SEEDS` / `S6E7_RUN_TAG` (train_common) and
  `S6E7_KEEP_WATER` (features). Bagged Day-2 arrays preserved as `*_bag`; champion arrays as
  `*_v1`; attribution set `*_bagw` (bagged, water kept) built for tomorrow.
- **`region_blend.py` — FLAT.** Separate blend+decision weights per missingness regime:
  +0.00002 plain / +0.00000 advwt vs global, and per-region blend weights ≈ global (~0.25 each).
  No differential regional model strengths; the missing-driver region is information-limited, not
  blend-limited. Below the >0.001 gate → not queued.
- Step-6 shift-aware MLP **skipped** — its gate (a validated test-like metric) failed.
- "Boundary-freeze hybrid" (v2 complete-region + v1b missing-region) considered and rejected by
  arithmetic: it would differ from v1b on only ~164 rows ≈ resubmitting the champion with noise.
- **Flip decomposition (offline, no LB reads): the WATER-DROP is the prime suspect for v2's
  regression.** Stage-wise test flips champion→v2: bagging 565 (75.8% missing-driver, 23.9% of the
  damaging 1,023), **water-drop 1,075 (88.3% missing-driver, 83.3% AR↔UNH, 62.2% of the damaging
  set)**, MLP+robust-weights 512 (73.4%, 24.0%). Paired OOF agrees (water-drop −0.00026 on same
  seeds/folds). Mechanism: in the near-zero-signal missing-driver region, removing even a noise
  column perturbs split tie-breaking → shifted the exact boundary the champion had right.
  Consequence: **`water_intake` is KEPT by default again** (features.py flipped: `S6E7_DROP_WATER=1`
  now opts INTO the drop; plain runs match the champion recipe); least-risky champion evolution =
  `attrib_bagw` (bagging only, water kept), if the user ever chooses to spend a submission.
  Caveat: flips≠errors — convergent circumstantial evidence, not proof.

### Day-2 evening: two independent advisor verdicts CONVERGE — the ceiling is proven
- **Missing-driver region 0.886 IS the Bayes ceiling** (probe-backed): explicit marginalization
  over the missing driver's marginal loses 2.1pp; proper conditional MI loses 3pp; per-pattern
  models/region-weighted losses/kernel tables all ≤ champion. Mechanism: features are NOT
  independent (sleep_quality proxies sleep_duration; step_count/exercise/calorie encode activity)
  and under MCAR trees directly estimate P(y|x_obs,pattern) = the Bayes target, proxies included.
  Five estimator families converge on 0.886; residual headroom ≤0.0008 overall. Scratch probes:
  /tmp/probe_bayes.py, probe_extended.py, probe_models.py (session-temporary).
- **Error-budget arithmetic**: +0.001 overall needs net +1,778 at-risk (or +119 fit / +173
  unhealthy) rows; cell oracles all sit BELOW champion; a PERFECT champion-vs-MLP per-row selector
  gains only +0.00177 → realistic model-diversity capture <0.0005. Decision-weight surface is
  flat (±20% weight noise costs 0.0002) — no more weight engineering in either direction.
- **LB noise measurement**: public-split sd ≈ 0.00147, private ≈ 0.00057. Every public delta ever
  observed here is sub-noise — the v1b>v2 "regression" was ~10 net rows; the water-drop damage
  attribution is downgraded to UNPROVEN (its flip-fingerprint analysis stands; the LB causality
  doesn't). Generator artifacts dead (quantized values, no clone linkage, z-std 0.97).
- **CONSEQUENCE — the game is private-LB variance, not levers**: (1) final #1 = champion v1b;
  (2) final #2 = breadth-averaged champion (8–10 seeds × **4 GBDT-family bases incl. hgbc**,
  water kept, NO MLP, v1b recipe);
  (3) seed-dataset citation check, 30-min hard cap (advisor found no schema match; no join key
  exists — expected dead); (4) optional ceiling-falsifier (predict champion errors given its own
  confidence; error-AUC ≤0.55 proves the row-level limit). Rejected en bloc: TabPFN/RealMLP/TabM
  as accuracy levers, linear_tree, RuleFit/EBM, kernel/cell methods, noise-robust/focal/weighted
  losses, region gating, more decision-weight work, artifact features, any public-LB-judged change.

### Day-3 (2026-07-03): every Tier-B probe VETOED by the transfer gate; breadth run stalled

All gates below are `diag_mlp_transfer.py` (need adv-weighted Δ>+0.001 AND test-like Δ>0 vs the
3-GBDT core). Full reports in `results/diag_*.txt`, `results/headroom.txt`.

- **`lgbm_ext` — external source data as train-only augmentation: VETOED.** Found and downloaded
  the seed dataset (`data/external/student_health_dataset_50k.csv`, exact 13-feature schema,
  near-identical class balance, ZERO missing values — almost certainly the generator's source).
  Concatenated into training folds only (`train_ext_aug.py`, S6E7_EXT_W=1.0). Solo −0.0007
  overall, −0.0015 in the missing-driver region; ensemble adv-wt Δ=+0.0000, test-like Δ=−0.0003.
  The clean rows teach the complete-driver regime, which is already solved. NOTE: only the
  *augmentation* use is dead; external↔competition row-linkage (clone matching) was never probed.
- **`lgbm_iw` — adversarial importance weighting: VETOED** (solo −0.0004; advwt Δ=+0.0001,
  test-like Δ=−0.0000). Confirms Tier-C prediction (problem is variance, not bias).
- **`lgbm_fe` — FE variant: worse** (weighted 0.9487, argmax −0.001 vs baseline). FE-saturation
  confirmed again.
- **`mlp_la` — logit-adjusted MLP retrain: VETOED** (0.9488; advwt Δ=+0.0000, test-like
  Δ=−0.0002; fixes 5.4% of core errors but 81.5% of fixes are in the missing-driver region →
  doesn't transfer, same failure mode as the original MLP).
- **`headroom.txt` — recombination is measured-exhausted**: a PERFECT per-row selector over
  {core, mlp_la, lgbm_ext} restricted to the transferable region (complete-driver ∩ test-like)
  gains **+0.0002** (221 rows). Upper bound over all rows +0.0039, but that lives where gains
  don't transfer.
- **`final2_breadth` (current CSV) is the 3-seed fallback**, blending the `_bagw` bags of
  lgbm/xgb/cat/hgbc (weights ≈0.25/0.26/0.22/0.27), OOF 0.9497. The intended 8-seed × 4-base
  breadth run (`run_breadth.sh`, reboot-resumable, skip-if-exists) completed only `lgbm_s42`
  before the box stopped (see [[project_pc_random_reboots]]) — **resuming it is the main open
  compute task**. `combine_breadth.py` rebuilds the blend from whatever seeds exist.

### Day-4 (2026-07-06 evening): clone-linkage probe — DEAD, with clean controls

`src/probe_linkage.py` → `results/probe_linkage.txt`. Question: are competition rows (noisy)
clones of the 50k source rows, such that matching back recovers missing drivers? **NO, three
independent ways:**
- **Exact**: 0 matches on the 7-numeric key (joint cardinality ~1e19) across all 727k
  complete-numeric competition rows; (step_count,bmi) pair-exactness at chance (71 vs ~56).
- **Noisy**: NN distances train/test→external ≈ a per-column-permuted null (p50 0.555 vs 0.557)
  — the numerics carry **no joint structure** linking competition rows to source rows; the
  uniform ~12% offset below ext-LOO spacing is a marginals artifact, not identity.
- **Usable recovery**: masked stress_level read off the matched external row = 0.340 accuracy,
  WORSE than train-self-match (0.360) and worse than majority (0.431); label likewise
  (0.799 < 0.811 < 0.857). External matching has no privileged information.

Consequence: the external dataset is fully exhausted (augmentation vetoed Day-3, linkage dead
Day-4; kNN-posterior features are bounded by the same result — matching carries less info than
the marginal). The 2026-07-07 plan's Track 2 is DONE; tomorrow = breadth run (Track 1),
ceiling-falsifier (Track 3), preservation (Track 4).

### ⚠️ PROTOCOL (user-set, 2026-07-02 pm): work LEADERBOARD-BLIND
The user watches the LB themselves; **Claude must not query submission scores**
(`kaggle competitions submissions`) or design LB-probing/attribution submission plans. Rationale:
optimizing against the public LB overfits the public split (final rank is private), and today's
own evidence shows neither OOF nor a few LB reads can rank <0.001 gaps. Consequences:
- **Discard any candidate whose expected effect is <0.001** — unrankable by any signal we allow
  ourselves, so never worth a submission. Only pursue plausibly-large levers (new signal,
  genuinely different model families, structural changes).
- Champion `ensemble_v1b` (reproducible: `S6E7_SEEDS=42 S6E7_KEEP_WATER=1`, non-bagged 3-GBDT,
  precorrected blend) is the standing submission; `submissions/champion_v1.csv` regenerates it.
- The Day-2 attribution CSVs (`attrib_bagw.csv`, `attrib_bag.csv`) exist on disk as insurance but
  are NOT queued — submitting them to explain a −0.0006 public delta is exactly the pattern the
  protocol forbids.
- Historical LB numbers below remain as context; they are not to be extended by new score reads.

| Model | OOF argmax | OOF decision-weighted | Public LB | Script |
|---|---|---|---|---|
| LGBM     | 0.9384 | 0.9488 | 0.94886 | `src/baseline.py` |
| XGBoost  | 0.9265 | 0.9483 | 0.94894 | `src/train_xgboost.py` |
| CatBoost | 0.9487 | 0.9491 | — | `src/train_catboost.py` |
| **Ensemble v1b** | 0.9493 | **0.9494** | **0.94970** | `src/ensemble.py` |

**The blend of the 3 GBDTs DOES lift** (+0.0008 LB) — but only once you blend on the *deployed*
(decision-corrected) surface. `ensemble.py::precorrect()` scales each model's probs by its own
saved per-class decision weights before the blend. ⚠️ **Do NOT blend raw probabilities under a
plain-argmax objective**: LGBM/XGB have poor plain argmax (0.938/0.927) and only become
competitive *after* decision correction, so an argmax blend objective discards them and collapses
to ≈100% CatBoost (an earlier v1 did exactly this → OOF 0.9491, LB 0.94861, *worse* than XGB alone).

**CV↔LB**: absolute calibration is excellent (LGBM 0.9488→0.94886, XGB 0.9483→0.94894, ensemble
0.9494→0.94970), but with adv-AUC 0.65 covariate shift OOF cannot *rank* models separated by
<0.001 (XGB's lower OOF beat CatBoost-heavy v1 on LB). Trust OOF for coarse decisions; confirm
near-ties on the LB.

---

## Modelling notes

- **CV**: 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the label. Scores →
  `results/cv_scores.csv`. All OOF/test arrays are `(n, 3)` probabilities in `CLASSES` order.
- **Imbalance / decision rule** (the primary lever for balanced accuracy): each model trains with
  class weighting (`CLASS_WEIGHT` toggle: LGBM `class_weight="balanced"`, XGB balanced
  `sample_weight`, CatBoost `auto_class_weights="Balanced"`), then `finalize()` tunes per-class
  **decision weights** on OOF (`argmax(proba·w)`, `kaggle_utils.tune_decision_weights`). Worth
  ~+0.010 over plain argmax here (e.g. LGBM 0.9384 → 0.9488). Both levers compound: class weights
  change what the trees learn; decision weights re-place the boundary on OOF.
- **GPU (RTX 2060, 6 GB)**: XGB uses `device="cuda"`. **CatBoost GPU has two 6 GB traps that make
  it silently fall back to CPU (~23 min/fold vs ~1 min GPU)**: (1) default `max_ctr_complexity=2`
  builds categorical feature-COMBINATION CTR tables → cap at **1**; (2) passing an `eval_set` holds
  a second CTR set on-GPU → **drop early stopping**, train fixed `ITERATIONS=1200` (val split still
  predicted for OOF). Also: the live GPU probe is unreliable right after a killed process, so
  CatBoost defaults `task_type="GPU"` directly (`CB_TASK_TYPE=CPU` to override). LGBM wheel is
  CPU-only (`lgbm_device.get_lgbm_device()` returns cpu).
- **GPU memory rule (any future PyTorch script)**: `del` model + optimizer + tensors before each
  fold returns; `torch.cuda.empty_cache()` after each fold/trial.

---

## Headroom analysis (4-agent brainstorm, 2026-07-02 — all probe-backed)

**The 3-GBDT stack is at the data's information ceiling.** A depth-8 tree on just
`{sleep_duration, stress_level, physical_activity_level}` scores 0.9490 ≈ the full 13-feature
ensemble; the label is a near-deterministic rule (`stress=medium`→99.4% at-risk; `stress=high &
sleep<6`→98.5% unhealthy, `≥6`→98.9% at-risk; `stress=low & active & sleep≥7`→fit) that the GBDTs
already fully absorb (OOF 0.9716 > hand-rule 0.9689 on complete-driver rows). **86% of remaining
error is in the 26% of rows with ≥1 key driver MISSING** (at-risk recall 0.763 there vs 0.972 when
present) — an MCAR information limit, not a modelling gap (missing values are unrecoverable;
`sleep_duration` corr ~0 with all features). `fit↔unhealthy` is already solved (~0.6% cross-conf);
the only bottleneck is `at-risk↔minority`.

## Next steps — GATED (do the cheap variance plays; most "levers" are proven flat)

**TIER A — do (cheap, low-risk):**
1. **Multi-seed bagging** — average OOF/test over 3× `StratifiedKFold` seeds per model. Fold-std
   ≈0.0013 > our last LB gain (0.0008): pure variance, cheap on GPU. Re-tune decision weights after.
2. **Regularize decision weights** — fold-average / bag the Nelder-Mead fit instead of one shot on
   full OOF (nested probe shows ~0.004 variance → OOF-overfit-prone). Near-free.
3. **Drop `water_intake`** — pure noise (identical class means) AND #3 adversarial-shift driver.
   Regenerate all OOF/test arrays + re-tune weights in one pass.
4. Keep **2 diversified final submissions** (hedge the CV↔LB rank noise).

**TIER B — one cheap probe each (genuinely additive but likely small; ceiling looks data-bound):**
5. **Focal-loss** custom objective (only loss idea not redundant w/ decision weights — reshapes
   within-class prob). 6. **Majority downsampling + recalibration** (frees tree capacity).
7. **Missingness-pattern × present-driver interactions** (targets the 26% error slice).

**TIER C — KILLED (proven flat / wrong-for-metric, don't build):** rule-encoding / monotonic score
(GBDT beats the rule); generic FE — interactions/ratios/sleep-debt/BMI-bands/missingness-count
(ablation −0.0005/−0.0002 = noise, FE-saturated for trees); logit-adjust / prior-correction /
effective-number weights / 3×3 cost matrix (subset of the decision-weight search); prior-matching
post-proc / pseudo-labeling / EM label-shift (shift is in P(x) not P(y|x), Δ<0.2pp); subset-specific
decision weights (tested flat); hierarchical 2-stage (fit↔unhealthy already solved); CV-scheme
overhaul / adversarial importance-weighting (problem is variance, not bias); Optuna sweep (tree
params are 2nd-order behind the decision rule here).

**DEFERRED (user parked "other models"):** non-GBDT family (MLP/TabM) — agents agree this is the
*actual* largest remaining lever (ensemble diversity), and the FE in Tier C only pays off as inputs
to it (ordinal encoding, stress×activity×sleep_quality interaction, fold-safe target encoding of
the 64-way combo). Revisit when ready to add model families.

---

## ▶▶ PLAN for 2026-07-07 (full day)

Priority order; Track 1 is background compute, start it first. All candidates face the standing
gate (adv-weighted Δ>+0.001 AND test-like Δ>0); everything else is descriptive-only, LB-blind.

1. **Track 1 — resume the 8-seed breadth run (background, ~4–5 h, reboot-resumable).**
   Two parallel shells: `src/run_breadth.sh lgbm hgbc` (CPU) and `src/run_breadth.sh xgboost
   catboost` (GPU) — skip-if-exists, so just rerun after any reboot. When done:
   `combine_breadth.py` → true 8-seed `final2_breadth`. Checks: OOF, flip-count vs champion,
   missing-region flip share. This finishes final #2, the last sanctioned deliverable.
2. ~~Track 2 — external↔competition CLONE-LINKAGE probe~~ **DONE 2026-07-06 evening: DEAD**
   (see Day-4 section above; `results/probe_linkage.txt`). External dataset fully exhausted.
3. **Track 3 — ceiling-falsifier (~1 h).** Predict champion OOF errors from features + champion
   confidence. Error-AUC ≤0.55 ⇒ row-level limit confirmed (ceiling verified, not asserted);
   materially higher ⇒ residual structure — inspect what it keys on before doing anything.
4. **Track 4 — preservation (afternoon, while Track 1 runs).** `playground-series-s6e7/` is
   entirely UNTRACKED in git — commit src/ + CLAUDE.md + README (data/, npy gitignored). Update
   this file + memory with the day's outcomes. Confirm the two finals: #1 `champion_v1.csv`,
   #2 8-seed `final2_breadth.csv`.

If Track 2 hits (linkage validates on train), it preempts everything: build the
impute-from-source pipeline, retrain the champion recipe on imputed drivers, re-gate.

## Experiments log

| Date | Run | Description | OOF balanced_acc |
|---|---|---|---|
| 2026-07-02 | lgbm | LGBM raw features, class_weight=balanced, + OOF decision weights | 0.9488 (argmax 0.9384) |
| 2026-07-02 | xgboost | XGB (GPU), balanced sample_weight, + OOF decision weights | 0.9483 (argmax 0.9265) |
| 2026-07-02 | catboost | CatBoost (GPU, max_ctr_complexity=1, no early-stop), Balanced, + decision weights | 0.9491 (argmax 0.9487) |
| 2026-07-02 | ensemble_v1 | NM blend on RAW probs, argmax obj → collapsed to 96% cat | 0.9491 (LB 0.94861) |
| 2026-07-02 | ensemble_v1b | blend on decision-CORRECTED probs (0.30/0.33/0.38) + tune | **0.9494 (LB 0.94970)** |
| 2026-07-02 | Tier A (bagged) | 3-seed bagging + robust(bagged) decision weights + drop water_intake. Per-model: lgbm 0.9489 / xgb 0.9484 / cat 0.9493 (all +0.0001–0.0002). Ensemble **0.9493** — FLAT vs 0.9494 (noise). Value = variance reduction: single vs bagged weights agree; argmax stability up (lgbm 0.9384→0.9408). | 0.9493 |
| 2026-07-02 | mlp | PyTorch MLP (cat embeddings + std numerics + missingness feats incl. driver-missingness), bagged 5-fold. Solo weighted **0.9489**; decision weights near-uniform & DIFFERENT from GBDTs (well-calibrated). Diverse (fixes 7.3% of lgbm errors ≈ catboost's 7.9%). | 0.9489 |
| 2026-07-02 | ensemble v2 | bagged lgbm/xgb/cat + MLP (0.23/0.25/0.28/0.24) + robust weights, water dropped. OOF **0.9495** (new best OOF) but **LB 0.94910 — REGRESSED** vs champion 0.94970. | 0.9495 (LB 0.94910) |
| 2026-07-02 | lgbm/xgb/cat `_bagw` | bagged 3 seeds, water KEPT (attribution set) | 0.9493 / 0.9488 / 0.9492 |
| 2026-07-02 | attrib_bagw / attrib_bag | v1b-recipe blends of the bagged GBDT cores (water kept / dropped). CSVs on disk, **not submitted** (LB-blind protocol). Descriptive only: water-kept 0.94955 vs water-dropped 0.94929 on OOF — a <0.001 gap, unrankable. | 0.94955 / 0.94929 |
| 2026-07-02 | hgbc_bagw | **NEW 4th base from a public notebook** (`train_hgbc.py`): sklearn HistGradientBoosting, early stopping scored on **balanced_accuracy** (the metric, not logloss), class_weight=balanced, stops ~35 iters, **~5s/fold on CPU**. Best single model; near-uniform decision weights (calibrated argmax). Joins final #2's breadth average. | **0.9494** |
| 2026-07-03 | mlp_la | Logit-adjusted MLP (`train_mlp_la.py`). Well-calibrated argmax but VETOED by transfer gate (advwt +0.0000, test-like −0.0002). | 0.9488 |
| 2026-07-03 | final2_breadth | Final #2 candidate: v1b-recipe blend of the four `_bagw` bases (3-seed fallback; 8-seed breadth run pending). `submissions/final2_breadth.csv`. | **0.9497** |
| 2026-07-03 | lgbm_ext | +50k external source rows as train-fold-only augmentation (`train_ext_aug.py`). VETOED (solo −0.0007; teaches the already-solved complete-driver regime). | 0.9485 |
| 2026-07-03 | lgbm_iw | Adversarial importance-weighted LGBM (`train_iw.py`). VETOED (advwt +0.0001). | 0.9489 |
| 2026-07-03 | lgbm_fe | FE variant (`train_fe.py`). Worse — FE-saturation confirmed. | 0.9487 |
| 2026-07-03 | headroom | Perfect-selector oracle over core+{mlp_la, lgbm_ext} in the transferable (complete ∩ test-like) region: **+0.0002** → base recombination measured-exhausted (`headroom.txt`). | — |

**LB**: lgbm 0.94886, xgboost 0.94894, **ensemble_v1b 0.94970 (best)**. The blend lifts once
models are pre-corrected to their deployed surface. Next lever for a bigger jump is a non-GBDT base
(the 3 GBDTs still only fix ~5–7% of each other's errors).

### Fold-score note
`cv_scores.csv` fold_* columns are plain-**argmax** balanced accuracy (fold variance), while the
`oof_balanced_acc` headline is the **decision-weighted** score — do not compare a fold column to the
headline directly.
