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

### Day-4 night (2026-07-06, user-directed missingness EDA): ⚡ MASK-MECHANISM SHIFT FOUND

`src/eda_missingness.py` → `results/eda_missingness.txt` + `results/figures/miss_*.png`.
The old "missingness ≈ MCAR" claim (from class-conditional null rates only) is FALSE in a way
that matters:

- **In TRAIN, four masks are deterministic single-column trigger rules** (hard zeros
  off-trigger): water_intake missing ⟺ gender=female (19.4%); physical_activity_level ⟺
  smoking_alcohol=occasional (16.1%); bmi ⟺ sleep_quality=good (6.2%); diet_type ⟺
  gender=other (3.0%). MAR-test AUCs 0.84–0.86 (`miss_mar_auc.png`); stress_level and
  sleep_duration masks are clean MCAR (AUC 0.50).
- **In TEST the SAME masks are UNIFORM** at the same marginal rates (e.g. water by gender:
  6.3%/6.3%/6.3% vs train 19.4%/0/0) — `miss_mechanism_shift.png`. Marginal rates match
  exactly, so rate-level checks never saw it.
- Consequences: (1) trees learn train-only NaN↔trigger couplings (water-NaN ⇒ female;
  activity-NaN ⇒ occasional-smoker — and activity is a KEY DRIVER) that misfire on ~2/3 of
  test NaN rows (~13% of test rows have ≥1 of the 4 columns missing); (2) **3,703 test rows
  (1.25%) have missingness patterns that never occur in train** (e.g. water+diet co-missing
  is a train-impossible gender contradiction); (3) this plausibly explains part of adv-AUC
  0.65 AND the water-drop flip fragility (water's NaN channel was a female flag in train —
  dropping the column deleted a gender proxy, moving the fragile boundary).
- Measured non-levers from the same session: indicators add ZERO about the drivers on
  train-mechanism rows (Δacc 0.0000 — triggers are observed columns); exact trigger-recovery
  (e.g. water-NaN ⇒ female when gender also masked) touches only ~0.5% of rows AND is
  test-invalid (test masks are uniform ⇒ NO recovery inference in test — do NOT wire
  train-rule recovery features; they'd inject WRONG values in test).
- Class dependence of bmi-mask (fit 2.9% vs unhealthy 0.7%, V=0.031) is fully explained by
  the sleep_quality=good trigger; patterns co-occur at independence otherwise.

**⇒ NEW TOP LEVER (tomorrow #0): missingness-mechanism repair.** (a) Quantify damage: take
OOF validation folds, RE-MASK them under the TEST mechanism (uniform at test rates), score
the champion — the delta estimates what the mechanism shift costs (also fixes our broken
evaluation: plain OOF validates under the WRONG missingness distribution). (b) Repair
candidates, gated: R1 impute-the-4 (model-impute water/bmi/diet/activity; identical treatment
train+test ⇒ mechanism difference vanishes); R2 uniform re-masking augmentation of train
(break the trigger coupling by masking the 4 columns in off-trigger rows too); R3 retrain
with the 4 columns' NaN semantics neutralized. Evaluate ALL candidates on the test-mechanism
re-masked OOF from (a), not plain OOF.

### Day-4 night part 2 (2026-07-06, user-directed): id-ordinality probe — target NULL, but masks are BLOCKWISE in test id

`src/eda_id.py` → `results/eda_id.txt` + `results/figures/id_*.png`. Ids: train 0–690,087,
test 690,088–985,841, both contiguous and file-sorted.

- **id carries NO target signal** (all measured): class rate per id decile chi2 p=0.019 with
  bin sds at binomial-noise level (`id_class_rates.png`); lag-1..10 target autocorrelation in
  id order = shuffled null, |z|<1.1 (`id_lag_agreement.png`); LGBM on id+id%k alone = bal-acc
  0.3328 ≈ chance. **Do NOT add id as a feature.**
- **Numeric features do not drift** over id, and test bins continue train means (max |z| ≈ 2.8
  over 50 bins = noise; `id_feature_drift.png`). Adv-AUC 0.65 is NOT id/segment drift.
- **⚡ But the TEST masks for water_intake and diet_type are BLOCKWISE in id**
  (`id_mask_regimes.png`, the money plot): long contiguous zero-missing runs (water: three
  ~5k-row blocks, 3.5% of test; diet: ELEVEN blocks, **67% of test has diet-mask OFF**, the
  rest bursts at 3–9%), with shared regime boundaries (both switch at test row 40,467). Train
  is stationary for all masks; the other 4 test masks are stationary too. So "test masks are
  uniform" (Day-4 part 1) refines to: **gender-independent everywhere** (inside active regions
  female/male/other = 7.1%/7.1%/7.2%) **but non-stationary in id** — the test mask generator
  ran in on/off segments.
- **Segments are NOT exploitable**: features, gender, and stress composition are identical
  across mask-active vs mask-zero segments (std-diffs ≤0.008), and trees never see id, so
  block structure ≈ iid at marginal rates from the model's viewpoint. Consequence for
  tomorrow's #0 instrument: **uniform re-masking at marginal rates remains the correct
  test-mechanism emulation** — no need to simulate blocks.

### Day-4 night part 3 (2026-07-07 00:xx): #0a DONE — mechanism-shift damage MEASURED, above gate

`src/diag_mask_shift.py` → `results/diag_mask_shift.txt`. Baseline-LGBM folds scored on three
val surfaces (weights fit on plain OOF, as deployed): plain 0.9487 / test-mechanism emulation
0.9467 / volume-matched control 0.9477 (weighted bacc). **Total emulated damage −0.0020,
splitting 50/50 into volume (−0.0010) and PLACEMENT (−0.0010, uniform vs on-trigger at equal
NaN count).** On rows receiving a new NaN (13.3%), bacc drops 0.9498→0.9347. Real-test
translation: test volume = train volume (marginals match) but ALL test NaNs are uniform while
the emulation could only uniformize half (existing train-mechanism NaNs can't be unmasked) ⇒
true cost ≈ −0.001..−0.002 — **clears the ≥0.001 gate; repair keeps #0.** R2 (uniform
re-mask augmentation in training) targets exactly the placement term; verify on this
instrument (rerun diag with the repaired model) BEFORE regenerating champion legs. Quick kills
same night: exact-duplicate/train↔test-match probe = ZERO everywhere (match-and-copy is dead);
champion's prediction mix on impossible-pattern rows is unremarkable; numerics are coarse
grids (water 400 uniques, heart_rate exactly 1dp, calorie/steps integer) → frequency/count
encodings are better-motivated for today's FE block.

### Day-5 (2026-07-07): R2a repair GATED PASS + wired in; three avenues measured-dead

**#0b — R2a repair (uniform re-mask of train) PASSES the instrument** (`src/diag_repair.py`
→ `results/diag_repair_r2a.txt`, exactly paired with the baseline diag: same folds, same val
remask RNG): testmech 0.9467→**0.9477 (+0.0011, gate ≥+0.001)**; repaired testmech = the
volume-matched control (0.9478) ⇒ the placement/mechanism damage is recovered **in full**;
plain surface +0.0001 (costless); newly-NaN'd rows 0.9347→0.9414. Weights-fit-on-testmech
adds only +0.0001 ⇒ the value is the retraining, not the weight surface.

**Wiring**: `S6E7_REPAIR=1` in `train_common.load_dataset()` applies `_uniform_remask` (the
4 `MECHANISM_SHIFTED` columns, iid Bernoulli at the TEST marginal rates, fixed RNG) to the
WHOLE train matrix — val rows included, so every base's OOF sits on the same test-mechanism
-emulated surface and decision weights are fit on it. **Repaired OOF reads ~0.001–0.002 below
plain-surface numbers by construction — only compare repaired to repaired.** Test is never
re-masked. `run_breadth.sh`/`combine_breadth.py` take `S6E7_TAG_PREFIX=_r` so repaired
artifacts (`*_r_s<seed>`) never mix with the unrepaired ones.

**Repaired legs (seed 42, weighted OOF on the repaired surface)**: lgbm 0.9478 / xgb 0.9473 /
cat 0.9479 → **`champion_repaired` blend 0.9484** (v1b recipe; weights ≈.33/.33/.35;
`submissions/champion_repaired.csv`). vs the unrepaired champion's estimated ~0.9474 on this
surface ⇒ +0.001, consistent with the instrument. Test-side: 566 flips vs `champion_v1.csv`
(0.19%), shifted-col-NaN rows enriched 2.8× (32% vs 11.3% base), dominant move
at-risk→unhealthy — mechanism-consistent, targeted, not a reshuffle. **User-directed
submissions (2026-07-07 pm): `champion_repaired.csv` + `lgbm_r_s42.csv` — LB-blind protocol
holds, user reads the scores.** Public-split sd ≈0.0015 ≈ the expected effect, so the public
delta is suggestive only; the repair's case rests on the gated instrument.

**Measured-dead today (all three properly gated, closing the 07-06 un-kills):**
- **3×3 cost matrix — FAIL** (`src/cost_matrix_probe.py` → `results/cost_matrix.txt`):
  in-sample +0.0000 over per-class weights (moves 0.125% of rows);
  `decision.split_half_gate` mean **−0.00003** (sd 0.00007) over 6 seed/half swaps. The
  un-kill was right procedurally, wrong empirically: per-class weights already saturate the
  decision layer on this blend.
- **HP tuning — FAIL** (`src/tune_hgbc.py` → `results/tune_hgbc.txt`, `optuna_hgbc.db`;
  deployed objective = decision-weighted bacc, repaired surface, incumbent pinned as trial 0):
  150 trials/90 min, best 0.9486 vs incumbent 0.9482 = **+0.0004** — and TPE converged (flat
  optimum: depth 7, lr≈0.2, ~32 leaves, min_leaf≈130 all within ±0.0002). hgbc was the family
  canary ⇒ no LGBM/XGB/Cat sweeps.
- **Driver-posterior features — FAIL** (`src/train_dp.py` → `lgbm_dp_r_s42`): 3 transductive
  aux LGBMs (stress/activity/sleep-bucket posteriors from train+test observed rows, label-free)
  +9 features. Solo 0.9476 vs 0.9478 identical-leg baseline (**−0.0002**); blend-level
  0.9485 vs 0.9484 (**+0.0001**, sub-gate; the NM giving dp the top weight is OOF-overfit on a
  near-tie). Root cause measured: the drivers are ~unrecoverable from other features (aux acc:
  stress 0.480 vs ~0.43 majority, sleep-bucket 0.557, activity 0.722) — same conclusion as the
  Day-2 marginalization probes, reached from the input side. `champion_repaired_dp.csv` NOT queued.
- **Frequency encoding — FAIL** (`src/train_freq.py` → `lgbm_freq_r_s42`): per-numeric value
  counts over train+test pooled + 64-way driver-combo count (the one FE family that injects
  dataset-level rather than row-wise information; motivated by the quantized-grid finding). Solo
  0.9479 vs 0.9478 (**+0.0001** — the only FE probe not strictly worse); blend-level 0.9485 vs
  0.9484 (+0.0001).
- **Exact-value TE of the numerics — FAIL but closest** (`src/train_te_num.py` →
  `lgbm_te_r_s42`; sourced from Mark Susol's public trail via the Mamarin notebook, claimed
  +0.0009): cross-fitted P(class | exact grid value) for the 6 informative numerics (nested
  inner 5-fold for train rows — invariant #3; NaN stays NaN; water excluded = noise). Solo
  0.9480 vs 0.9478 (**+0.0002**); blend-level **0.9487 vs 0.9484 (+0.0003, the day's largest
  blend delta, TE leg got top NM weight)** — still 3× under the gate, NOT queued. Susol's
  effect size does not replicate against a decision-weighted repaired stack.
  **FE is now fully measured: 6 families (row-wise transforms, rule-combo target encoding,
  missingness indicators, driver posteriors, frequency stats, exact-value TE), 6 sub-gate
  results, one mechanism** — an axis-aligned 3-feature label rule the trees already saturate,
  with the residual error in deleted-information rows. Ranked residue if ever revisited:
  TE (+0.0003 blend) > freq (+0.0001) > dp (+0.0001).

**Notebook scrape (Mamarin, "quit chasing 0.950", 2026-07-07)**: independently confirms the
wall thesis (4 model families within 0.0006; MLP blend +0.0001 at 99.1% agreement; public
mirage per S6E6 shakeup), prior-correction b≈1 ≡ our decision weights ("weights in training
and prior-correction are substitutes — stacking both fixed costs −0.045"), Kawamata measured
value-snapping dead + external-50k harmful (matches our Day-3), nybbler's ceiling decomposition
matches ours. Their missingness read is rate-level MCAR — the mechanism shift (our repair) is
NOT in the public material. Reference floors: chance 0.3333 / stress-only rule ≈0.85 / naive
argmax GBDT 0.8783 / decision-corrected anything ≈0.949 — regression alarm = any run <0.9488.

**Day-5 wrap**: breadth run complete, 8 seeds × 4 bases, all repaired. Per-base combined
weighted OOF (repaired surface): lgbm 0.9484 / hgbc 0.9484 / xgb 0.9481 / cat 0.9481 →
**`final2_breadth_r` blend 0.9487** (weights ≈.25/.22/.27/.26, `submissions/final2_breadth_r.csv`)
— the repaired-lineage final #2. Several user-directed curiosity submissions went to the LB
today; per the user's instruction their scores are deliberately NOT recorded here or in memory.
Finals shape unchanged: #1 v1b lineage, #2 repaired lineage; the repair's case rests on the
instrument, not public reads.

### Day-6 (2026-07-08, ~2h session): gate FIXED + the crux re-gate — repaired MLP VETOED, signature unchanged

**Rung 0 — the transfer gate is now trustworthy** (`src/diag_mlp_transfer.py`, rewritten):
- Comparator core = the DEPLOYED repaired breadth blend `oof_ensemble_r_breadth.npy`
  (env-override `S6E7_CORE`), loaded as a raw array — it is already precorrected+NM-blended
  by `combine_breadth.py` and its weights JSON is nested (must NOT re-enter
  `precorrected_blend`).
- Adversarial scores cached PER SURFACE (`adv_scores_train_r.npy` under `S6E7_REPAIR=1`);
  the old shape-only cache bug is dead. The gate run itself must execute under
  `S6E7_REPAIR=1` (region masks + adv scores come from `load_dataset()`); `main()`
  hard-fails on core/surface mismatch. **Surfaces must match** — plain-surface candidates
  cannot be gated against the repaired core (confounded); compare signatures across days,
  not decimals.
- Candidate joins as a fixed mixture `(1-w)·core + w·candidate` at **w=0.20** (≈ a 5th
  family joining a 4-family blend — the old gate's marginal-member semantics; NM weights
  overfit near-ties, equal weight would inflate lift), with a w∈{.1,.2,.3,.5} sensitivity
  grid printed. Error-overlap decorrelation stats added per the exploratory ruling.
- Sanity (DP2, `results/diag_gate_sanity.txt`): core overall weighted bacc **0.9487** =
  Day-5 `final2_breadth_r` exactly; core member `hgbc_r_breadth` shows solo Δ −0.0003,
  fix-share 2.3%, error-overlap 92.9% — the instrument behaves.
- **⚡ Repaired-surface adv-AUC = 0.6886, HIGHER than plain 0.65** (we expected lower).
  Mechanism: the repair adds uniform NaNs but cannot unmask the original trigger-coupled
  ones, so the 4 shifted columns carry MORE total NaN in train than test — the adversarial
  classifier partly keys on that excess. The instrument stays internally consistent
  ("test-like" = most test-resembling under the deployed surface), but note the repair
  itself bakes in a residual train/test asymmetry. Runtime note: each `wbacc` = 8
  bootstrap-bagged NM fits on 690k rows → a full diag report is ~48 NM fits, tens of
  minutes; don't widen W_GRID casually.

**Rung 1 — the crux falsifier came back FLAT with a trustworthy instrument.**
`S6E7_REPAIR=1 S6E7_RUN_TAG=_r` runs of the existing trainers (3 seeds × 5 folds each):
`mlp_r` weighted OOF **0.9476**, `mlp_la_r` **0.9476** (repaired surface; GBDT legs sit
0.9478–0.9482). Gate on `mlp_r` (`results/diag_mlp_r.txt`): **VETO** — advwt Δ −0.0000,
test-like Δ −0.0001 at w=0.20, flat-to-negative at every w (w=0.30 loses −0.0001 across
the board: more MLP weight = strictly worse). The decorrelation signature is UNCHANGED
from the pre-repair Day-2 MLP: fix-share **5.1%** (Day-2: 5.1%), fixes **27.0% test-like /
90.2% missing-driver** (Day-2: 27.7% / 82.5%), and error-overlap with the core **92.3%
overall / 92.8% test-like — the same as a GBDT core member (92.9%)**. The mask-mechanism
repair did NOT unlock transferable NN diversity: the MLP is decorrelated only where the
information doesn't exist. (`mlp_la_r` gate left running unattended at session end —
report at `results/diag_mlp_la_r.txt`; expected VETO given its identical 0.9476 OOF.
Verify next session before citing.)

**Consequence**: the reopened avenue closes as MEASURED this time, not asserted. Per the
07-07 exploratory ruling, **Rung 2 (setenc / mask-consistency / DANN) starts any future NN
session DEMOTED, not dead** — their invariance mechanisms are structurally different from a
one-shot repair, but the cheap falsifier is flat, so they need a prior reason to expect a
different signature (e.g. setenc's structural inability to learn NaN↔trigger couplings)
before spending GPU time. The finals shape is unchanged: #1 v1b lineage, #2 repaired
lineage (`final2_breadth_r`).

### Day-7 (2026-07-09/10): overshoot mult sweep — repair VALIDATED, mult 0.5 marginally better (sub-gate, no rebuild); TabPFN priced DEAD; mlp_la_r VETO confirmed

**1. The Day-6 adv-AUC anomaly is resolved: the repair survives a volume-honest instrument.**
Train/test marginal NaN rates are ~EQUAL per column, so R2a (re-mask at full test rates on
top of trigger NaNs) ~DOUBLES train NaN volume in the 4 columns (water 0.122 vs test 0.063,
activity 0.103/0.053, bmi 0.040/0.020, diet 0.020/0.010) — and the testmech val surface
overshoots identically, so the Day-5 gate was measured on a surface sharing the flaw.
New instrument in `diag_repair.py`: **TESTVOL** = the testmech OOF restricted to val rows
complete in the 4 columns (86.0%, 593,550 rows) — on those rows the re-mask IS the true test
surface (uniform mechanism at exact test volume); biased toward the non-triggered population
but identical across runs (paired, fixed RNG streams). Sweep via `S6E7_REPAIR_MULT` (now also
wired into `train_common._uniform_remask`, default 1.0), reports
`results/diag_repair_r2a_tv_m{100,050,000}.txt`:

| training repair | testvol | testmech | plain |
|---|---|---|---|
| m000 none       | 0.9476 | 0.9468 | 0.9487 |
| m100 deployed   | 0.9484 | 0.9476 | 0.9488 |
| m050 half       | **0.9486** | **0.9478** | **0.9489** |

Conclusions: (a) **the repair is real** — +0.0008 testvol over no-repair; the coupling damage
is genuine, `final2_breadth_r` stands validated; (b) **mult 0.5 ≥ mult 1.0 on every surface
but only by ~+0.0002** — sub-gate, does NOT justify a breadth rebuild alone; if a rebuild
ever happens for another reason, set `S6E7_REPAIR_MULT=0.5` for free. Repro note: the m100
rerun reads testmech +0.0009 vs Day-5's +0.0011 — the 0.001 gate line flickers on rerun
noise; the mult comparison is paired within-chain and unaffected. adv-AUC on the mult-0.5
surface: **0.6704** (`results/adv_auc_r_m050.txt`) — between plain 0.65 and m100's 0.6886,
corroborating the volume mechanism (less overshoot → surface closer to test).
m000 report note: killed after table 1 (redundant NM tail); its TESTVOL computed post-hoc
from the saved OOF arrays and appended.

**2. TabPFN-v2 missing-region specialist: PRICED, DEAD** (`src/probe_tabpfn.py`,
`results/probe_tabpfn.txt`). Feasible on the 6 GB card (peak 2.17 GiB, fit 14.5s, predict
1.9s/1k rows ≈ 7 min per full region-OOF context) — but on 20k held-out missing-driver rows
(10k context, repaired surface) it scores 0.8827 WITH in-sample-fit decision weights (a
generous upper bound) vs the deployed core's 0.8970 on the same rows, per-class recalls
strictly dominated. −0.014 in-region with no complementary error structure → no blend case;
the transfer gate would veto. Consistent with the deleted-information mechanism. NOTE:
current `tabpfn` 8.x (TabPFN-2.5) is license-gated (needs `TABPFN_TOKEN` from
ux.priorlabs.ai); the probe used `tabpfn==2.2.1` in an isolated venv because it pins
sklearn <1.7 — ⚠️ installing it into the main venv silently downgrades sklearn (happened,
reverted). The "still live" list from Day-5/6 is now empty of model-family levers.

**3. mlp_la_r gate rerun (Day-6 loose end): VETO**, as predicted (`results/diag_mlp_la_r.txt`):
advwt Δ +0.0000 / test-like Δ +0.0001 at w=0.20; fix-share 4.6%, 88.3% of fixes
missing-driver, error-overlap 92.2% — the same signature as mlp_r. NN closure is now
measured for both repaired variants.

**4. Infra (iteration-speed rule, user-set)**: `train_common.robust_decision_weights` bags
now run in parallel processes (bit-identical results; the sequential version burned ~30-60
min per call with 11 cores idle — it was the wall-clock bottleneck of every diag/combine
run). Before queueing any multi-run chain, estimate per-run wall-clock and parallelize the
bottleneck first; run the decisive comparison earliest in the chain.

**Finals unchanged: #1 `champion_v1.csv` (v1b), #2 `final2_breadth_r.csv` — now with the
repair's instrument-anomaly resolved in its favor.**

### Day-8 (2026-07-10): ARCHITECTURE ZOO — ⚡ RealMLP-TD + TE recipe BREAKS the solo ceiling; FINAL #2 SWAPPED to `final3_realmlp`

**The user's mandate: stop consolidating, explore architectures.** Six families run
through a shared harness (`src/zoo_common.py`: single-seed 5-fold `zoo_cv` with per-fold
reboot checkpoints + `te_block_for_fold`, the cached deferred-for-NN input recipe =
exact-value TE of 6 numerics + rule-combo TE + 4 ordinals, inner-cross-fit per invariant
#3, cache keyed (surface, seed, fold)). All on the repaired surface, gateable vs the
deployed core `ensemble_r_breadth` (0.9487).

**Zoo scoreboard (repaired surface, weighted OOF; signature = fix-share / %fixes
missing-driver / error-overlap):**

| cand | OOF | signature | verdict |
|---|---|---|---|
| **realmlp** (RealMLP-TD + TE recipe, `train_realmlp_td.py`) | **0.9492/seed** (s42=s7=s123 — seed-robust) | 5.9% / 77.6% / **94.5%, but solo BEATS core on EVERY subset** (+0.0005 overall, +0.0005 test-like, +0.0010 miss-driver) | formal w=0.2 gate veto (+0.0003) — **superseded by blend evidence below** |
| ftt (FT-Transformer, ABSENT missing tokens = setenc-lite, `train_ftt.py`) | 0.9478 | 3.8% / 77.1% / 91.9% | VETO — **the mlp_r signature; the Rung-2A structural prior is measured-dead in its cheap form** |
| rf (`train_trees.py`) | 0.9480 | 4.7% / 80.3% / 95.0% | VETO (core-member-like, weaker) |
| extratrees | 0.9219 | 23% / **97.7%** / 50.9% | VETO — maximally decorrelated exactly where information doesn't exist |
| tabm (pytabkit, `train_tabm.py`) | (ran evening — see log) | | |
| dann+mask-consistency (`train_dann.py`, built+smoke-tested) | NOT RUN (GPU budget) — queued overnight | | |

**⚡ THE FIND — `realmlp`: RealMLP-TD (package `realmlp.py`, balanced-softmax
`loss_prior_power=1.075`, train_bs=512+AMP+fused) fed the TE input recipe.** Mechanism
as predicted by the Rung-2-parallel note: TE posteriors convert the discontinuous
3-feature rule into a near-linear map; the s6e6 lesson ("the break came from porting
STRONG bases; RealMLP gave +0.0019") replicates here at this dataset's scale. NOT the
mlp_r pattern: improvement is UNIFORM across regions (transfers), error-overlap high
(94.5%) = mostly-nested-but-stronger, and fixed-w core+realmlp mixtures improve
MONOTONICALLY to w=0.5 on plain/test-like/advwt (no NM fitting — cannot be
blend-overfit). Three seeds all 0.9492 weighted.

**Blend + gates → the finals swap:**
- `final3_realmlp` = `blend_named` over {lgbm,hgbc,xgboost,catboost}`_r_breadth` +
  `realmlp_r_breadth` (3-seed avg 0.9493): NM collapses onto realmlp (**w=0.845**),
  OOF **0.9494** (repaired surface; incumbent final #2 = 0.9487).
- **Split-half blend gate** (`probe_blend_gate.py`, NEW: NM + decision weights fit on
  half rows, scored on holdout, 6 splits x 2 directions): **12/12 cells positive, mean
  +0.00042** — real, not NM overfit; below the +0.0005 submission-queue bar.
- **Volume-honest subset** (complete-in-4 on the repaired matrix, paired):
  all +0.0007 / complete4 +0.0006 / miss4 +0.0008 vs incumbent — sign-consistent.
- **⇒ FINALS: #1 `champion_v1.csv` (v1b, unchanged); #2 = `submissions/final3_realmlp.csv`**
  (CV-decided on three concordant instruments; LB-blind protocol intact — curiosity
  submissions of realmlp_r_s42/rf_r_s42/ftt_r_s42/final3_realmlp went up user-directed,
  scores unrecorded).

**Also measured today:**
- **Error-AUC ceiling falsifier** (`probe_error_auc.py`, Day-2 item 4 finally run):
  predicting core OOF errors — conf-only AUC 0.9538, features+conf 0.9560, **increment
  +0.0021 ≈ nothing** ⇒ row-level limit independently corroborated. (realmlp's +0.0005
  is boundary calibration everywhere, not error-picking — consistent.)
- **mult0.5+TE stacks** (`diag_repair.py r2a_te`, paired chain): testvol
  0.9476(m000)/0.9484(m100)/0.9486(m050)/**0.9488(m050+TE)**; +0.0004 vs deployed m100 =
  at the pre-registered rebuild gate ⇒ `_r2` rebuild (mult 0.5) justified; runs
  overnight (xgb s42 done; skip-if-exists). NOTE: realmlp already carries the TE block
  as inputs, so the blend delivers the TE residue through a stronger vehicle — TE-in-
  GBDT-trainers wiring deferred unless the `_r2`+realmlp reblend wants it.
- Seed plumbing fix: zoo scripts originally hardcoded `zoo_cv(seed=42)` and ignored
  `S6E7_SEEDS` — caught because a "seed 7" run reproduced seed-42 folds bit-identically;
  all 5 zoo trainers now use `SEEDS[0]` + seed-tagged checkpoints.

**Tomorrow's open items:** overnight `_r2` lane verdict (reblend `_r2` bases + realmlp,
paired vs final3); DANN/mask-consistency run + gate (built, smoke-tested); TabM verdict
if timeboxed out tonight; more realmlp seeds (marginal but free GPU); consider
realmlp-recipe variants ONLY if a mechanism argues a different signature (no HP fishing).

### Day-8 EVENING (2026-07-10): COMBINER UPGRADE — the s6e6 LR stacker replicates; FINAL #2 → `metablend_r`

User asked whether s6e6's best combiner (NOT NM) applies here — it does. All three
combiners run over the same strong-5 bases ({4 GBDT}`_r_breadth` + `realmlp_r_breadth`):

- **`build_lr_stack.py` (NEW, port)**: `data_science_stuff.kaggle.stacking.stack_oof`
  (per-(model,class) logit columns → multinomial LR C=1.0 balanced, honest 5-seed × 5-fold
  outer cross-fit) + `robust_decision_weights` after. Both variants identical — raw
  0.9494/0.9513/0.9442 (weighted/complete4/miss4), precorrected 0.9494/0.9512/0.9443:
  the LR meta relearns any per-class pre-scaling, so precorrection is irrelevant to it.
  ⚠️ Face-value tie with `final3_realmlp` (0.9494) is NOT a tie: the stack OOF is
  honestly cross-fitted, final3's NM weights are full-fit-optimistic.
- **`probe_combiner_gate.py` (NEW)**: the honest ranking — 12 split-half cells, each
  combiner + decision layer refit on half A, scored on half B. **LR−NM +0.00032
  (10/12 positive); metablend−NM +0.00032 (11/12, lowest sd); metablend−LR ±0.00000.**
  NM's scalar-per-model limit is real here, worth ~+0.0003 — the s6e6 pattern at this
  dataset's compressed scale.
- **`run_caruana.py` (NEW)**: greedy selection over ALL 12 repaired legs (incl. vetoed
  zoo/FE legs) — honest holdout 0.94938 ≈ tie with LR; picks = **realmlp 62%**, hgbc 12%,
  mlp_la_r 10% (only vetoed leg earning a seat; noise-level), rest scraps. Third
  independent confirmation of realmlp dominance; Caruana adds nothing over LR.
- **`metablend_r` artifact** = uniform avg(final3_realmlp, lrstack_r) probs + decision
  weights: 0.9494 / complete4 0.9512 / **miss4 0.9443 (best)**.
- **⇒ FINAL #2 = `submissions/metablend_r.csv`** (gate-preferred on sign-consistency
  11/12; s6e6 precedent — its best LB was exactly this avg-of-two-metas shape). #1
  unchanged (`champion_v1`). Curiosity submissions (user-directed, scores unrecorded):
  lrstack_r, final2_breadth_r (incumbent calibration read), metablend_r — daily cap is
  10, not 5.
- **Overnight re-armed** (`overnight_day8.sh`): `_r2` xgb/cat remainder → **realmlp
  `_r2` (mult 0.5) × 3 seeds** (SURFACE_TAG now env-aware: TE cache tag "r2" under
  mult≠1) → DANN → TabM retry; CPU lane lgbm/hgbc `_r2`; final combine includes
  realmlp. Tomorrow's decisive comparison: all-m050 candidate × best combiner vs
  `metablend_r` (never mix m100/m050 arrays in one combiner — surface rule).

### Day-8 LATE (2026-07-10, 16:00–21:00): combiner tournament, NN-closure completed, the m050 lineage priced DEAD at blend level

**Combiner tournament** (`probe_combiner_gate.py` generalized to 8 arms, 12 split-half
cells; `build_alt_meta.py` NEW = s6e6 ridgecal-GBDT + NN metas ported): every meta beats
scalar NM; **winner `lr6` = LR-on-logits + `mlp_la_r` as 6th base, +0.00040 (11/12)** —
the twice-vetoed leg earns a real seat through per-class weighting, exactly as Caruana
hinted. Tree/NN metas positive but lose to LR (+0.00016-18: no regional trust structure
to exploit, consistent with region_blend FLAT). lr6 full honest build (`lrstack6_r`)
**0.9495** = best m100 number; but only +0.00008 over `metablend_r` → below the +0.0003
displacement gate ⇒ **final #2 stays `metablend_r`**. `read_lineage.py` NEW = the
cross-lineage instrument: m050 masks are NESTED subsets of m100 masks (same RNG stream),
so the intersection = complete-in-4-under-m100 (510,866 rows) with identical val inputs
across lineages.

**DANN + mask-consistency: VETO — NN closure is now TOTAL.** Solo 0.9472 (invariance
tax vs mlp_r 0.9476); signature: fixes **95.2% missing-driver**, overlap 91.7% — the
same fingerprint as every NN. Four mechanisms measured (plain, logit-adjust,
absent-token attention, learned domain-invariance + dual-mask consistency): ALL
concentrate their diversity where information was deleted. Rung 2 is closed by
measurement across its whole design space.

**The m050 (`_r2`) lineage: DEAD at the deployed level.** Full rebuild completed (4
GBDT × 8 seeds + realmlp × 3, mult 0.5). Own-surface numbers read +0.0005 high (less
remasking = easier surface — inflation, not lift): `ensemble_r2_gbdt` 0.9492 own /
**0.9505 intersection vs m100 core 0.9506** (wash); `final5r2_r` (lr5 stack over the
_r2 strong five) 0.9499 own / **0.9512 intersection = exact tie with metablend_r**.
The Day-7 "mult 0.5 sub-gate, no rebuild" call is confirmed end-to-end; the diag
chain's +0.0002 does not survive to deployed blends. Do NOT chase m050 further.

**FINALS (updated 07-11 am): #1 `champion_v1.csv` · #2 `lrstack6_r.csv`** (was
metablend_r for ~12h). Basis: lrstack6 leads every honest instrument — best honest OOF
(0.9495), combiner-tournament best mean (+0.00040, 11/12), best intersection read
(0.9513) — and a user-directed 07-11 LB read broke the metablend-vs-lr6 near-tie the
same direction (paired deltas between near-identical candidates are low-variance;
metablend publicly underperformed BOTH its parents; scores themselves unrecorded per
protocol). #1 stays the realmlp-free hedge. All 10 daily curiosity submissions were
used 07-10. TabM retry + overnight tail completed (see log).

**▶▶ TOMORROW #1 (user-flagged, 2026-07-10 night): CHAINED BINARY CASCADE — the last
assertion-kill standing.** s6e6's `chain_cascade`/`chain_cascade_xgb` (binary stages,
multiplicative recombination via `data_science_stuff.kaggle.decision`) earned seats in
its winning LR stack. Here "hierarchical 2-stage" was Tier-C-killed by the reasoning
"fit↔unhealthy already solved" — but that evaluates the WRONG stage. The case: **stage 1
= at-risk vs minority as a dedicated BINARY problem** attacks the one live boundary (79%
of consequential flips; all residual error is at-risk↔minority placement), and a binary
objective changes what the trees LEARN — not covered by the cost-matrix kill (decision
layer only, measured dead) for the same reason class weights ≠ decision weights.
Build `train_cascade.py`: stage 1 lgbm at-risk-vs-rest (class-weighted binary), stage 2
fit-vs-unhealthy on minority rows, recombine P(c) = P(stage1)·P(c|minority); through
`zoo_cv`/`finalize` on the m100 surface (S6E7_REPAIR=1, tag `_r_s42`), ~30 min CPU.
Gates: diag signature + add as base to the LR-combiner tournament (the per-class meta is
the consumer that extracted value from mlp_la_r where scalar NM couldn't). Honest EV:
solo ~0.948x like everything; the question is differently-placed errors ON the live
boundary. Expected verdict either way closes the last un-measured Tier-C kill.

**▶▶ TOMORROW #2 (user idea, 07-10 night): EXACT-RULE DEDUCTION FEATURES.**
discussion/717222 (broccoli beef) proved the original's label is an EXACT depth-4 tree:
sleep<6 → (stress=high → unh, else ar); sleep≥6 → (stress≠low → ar; stress=low &
act≠active → ar; active & sleep≥7 → fit, sleep∈[6,7) → ar). **Two dual sleep thresholds
(6 AND 7).** Built tonight: `rule_features.py` = THREE-VALUED DEDUCTION — per row, the
set of labels reachable over completions of missing drivers (6 distinct sets;
stress=medium ALONE forces at-risk). This is the one thing prior FE lacked: a NOISE-FREE
logical prior on partially-missing rows, aimed exactly at the at-risk↔minority boundary.
Marginalization/rule-encoding kills DON'T cover it (those replaced predictions or
targeted complete rows). lgbm canary (`train_rule.py`) launched tonight — read
vs lgbm_r_s42 0.9478 paired. If ≥ +0.0002: feed rule_set to REALMLP (one-hot; the
NN consumer is where the recipe pays) + fix the combo TE triple — **audit finding:
train_fe/zoo TE used sleep_QUALITY; the true rule triple is sleep_DURATION-buckets
(6/7) × stress × activity** — never measured with the right third feature.

**▶▶ TOMORROW #3 (user idea, 07-10 night, lower priority): TABM FROM SCRATCH.** The
0.9411 verdict was config, not architecture (no loss hook). A pure-PyTorch TabM is
small: shared MLP backbone + BatchEnsemble rank-1 per-member adapters (r,s vectors) +
per-member biases; reuse `models.losses.smooth_ce_loss` (balanced softmax — the knob
that made realmlp work), fp16 AMP + GradScaler (FTT pattern), fused AdamW, torch.compile
(triton 3.2 present), whole-dataset GPU tensors, optionally PBLD embeddings from
`realmlp.py`. ~5 min/fold est. Honest EV: same family slot as realmlp (expect ±0.001,
high overlap); value = one more leg for the LR combiner, which extracts whiskers from
near-redundant legs (mlp_la_r precedent). Run AFTER cascade + rule-realmlp.

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
effective-number weights (subset of the decision-weight search); prior-matching
post-proc / pseudo-labeling / EM label-shift (shift is in P(x) not P(y|x), Δ<0.2pp); subset-specific
decision weights (tested flat); hierarchical 2-stage (fit↔unhealthy already solved); CV-scheme
overhaul / adversarial importance-weighting (problem is variance, not bias).

**⚠️ UN-KILLED 2026-07-06 (kill-audit — these were asserted, never measured; see the 07-07 plan):**
the **3×3 cost matrix** ("subset of decision weights" was wrong math — it strictly generalizes
them) and the **Optuna/HP sweep** (zero trials were ever run; "2nd-order" was advisor reasoning).

**DEFERRED (user parked "other models"):** non-GBDT family (MLP/TabM) — agents agree this is the
*actual* largest remaining lever (ensemble diversity), and the FE in Tier C only pays off as inputs
to it (ordinal encoding, stress×activity×sleep_quality interaction, fold-safe target encoding of
the 64-way combo). Revisit when ready to add model families.

---

## ▶▶ PLAN for 2026-07-08 — OUTCOMES (see Day-6 section): RUNG 0 **DONE** (gate fixed,
## sanity-passed; repaired adv-AUC 0.6886 caveat); RUNG 1 **VETO** (mlp_r + mlp_la_r both
## 0.9476, signature identical to pre-repair — no transferable diversity); RUNG 2 not run
## (2h budget) and now DEMOTED per the falsifier logic below. Original plan follows.

## PLAN for 2026-07-08 — NEURAL-NET DAY (diversity, not accuracy)

**Framing**: the 3-GBDT stack is at the Bayes ceiling; a NN cannot raise solo accuracy. The
ONLY value a NN adds is *decorrelated error that TRANSFERS to test-like rows*. Every prior
MLP (`mlp` Day-2, `mlp_la` Day-3) FAILED the transfer gate — **BUT all those verdicts are
PRE-REPAIR** (measured vs the un-repaired core, before `S6E7_REPAIR=1` existed, Day-5). The
repair uniform-remasks `physical_activity_level` (a KEY DRIVER) — exactly the missing-driver
region where the MLP's non-transferring diversity lived. **The crux re-gate on the repaired
surface has never been run.** That reopened avenue is the day. (3-agent brainstorm 2026-07-07;
all three converged on this + the gate bug below.)

**⚠️ RUNG 0 — FIX THE GATE FIRST (load-bearing; NOT a one-liner)**. Both parts required
before any NN result is trusted:
1. **Repoint the comparator core to the REPAIRED legs.** `adv_eval.py:27` (`GBDTS`) + line
   100 and `diag_mlp_transfer.py:51` load unrepaired `oof_{lgbm,xgboost,catboost}.npy`. Gate
   against the DEPLOYED 8-seed `_r` breadth blend (`oof_ensemble_r_breadth.npy`) — that's what
   a new NN would actually ensemble with.
2. **Regenerate `adv_scores_train.npy`.** `diag_mlp_transfer.load_adv` keys the cache on SHAPE
   ONLY, so under repair it silently serves the stale unrepaired scores — and the test-like
   subset is DEFINED by those scores. The repair changes train's missingness distribution ⇒
   which rows are "test-like" changes. Regenerate on a documented surface (or freeze the mask
   from unrepaired features and apply identically to both sides — pick one, document it).

**RUNG 1 — cheapest falsifier**: run existing `train_mlp.py` + `train_mlp_la.py` with
`S6E7_REPAIR=1`, re-gate vs the repaired core. Flag-flip, ~zero new code. Repaired MLP MOVES
on test-like rows → avenue reopens. Ties → weak (not conclusive); the mechanism-targeting
ideas below add invariance a one-shot repair can't, so proceed anyway.

**RUNG 2 — the two standouts (attack the NaN↔trigger coupling from OPPOSITE sides)**:
- **A. Present-only set/attention encoder** (`train_setenc.py`, NEW): row = set of present
  (feature,value) tokens; missing = token ABSENT (no sentinel, no indicator). **Structurally
  CANNOT learn the train-only NaN↔trigger coupling** (water-NaN⇒female etc.) that survives
  even the repair (repair adds NaNs, can't unmask existing ones) ⇒ compounds with, not
  duplicates, the repair. ≤13 tokens, tiny, <6 GB. New backbone (RealMLP's `NumericalPreprocessor`
  chokes on NaN + has no mask channel).
- **B. Mask-consistency co-training** (on `train_mlp.py`'s `NNData` backbone): two independent
  masks per row (iid Bernoulli at TEST rates), KL/JS consistency penalty, **SEMI-SUPERVISED
  over the 295k unlabeled test rows** (transductive, leak-free). Penalizes routing through any
  feature's presence ⇒ forbids the coupling by *invariance* (which the one-shot repair can't
  impose). λ needs a small sweep. Train on fresh masks off the RAW matrix, score OOF on the
  repaired surface — do NOT double-mask on top of `S6E7_REPAIR`.
- **C. Gradient-reversal domain head (DANN, USER-PITCHED 07-07 evening)**: aux head predicts
  train-vs-test from the trunk representation, gradient REVERSED into the trunk (λ ramp-up
  schedule) ⇒ representation is stripped of everything domain-informative — the LEARNED
  version of the invariance B imposes by construction, fit on the real shift (adv-AUC 0.65 +
  mask mechanism). Semi-supervised over the 295k test rows (label-free, leak-free). NOT the
  `lgbm_iw` kill: IW re-weights rows (bias fix); DANN changes the representation — and the
  veto predates the mask-shift discovery. Risk: the domain signal IS partly the NaN channel,
  which is weakly label-informative — λ too high strips label signal. Train on the RAW matrix
  (repair would hide the shift the head must see). **B and C share one backbone + unlabeled
  test loader — build as one script with two optional loss terms.**

**RUNG 2-parallel — the user's low-hanging fruit (INPUT RECIPE, orthogonal)**: feed a NN the
features trees don't need. A tree gets the 3-way AND / per-grid posterior FREE from splits
(→ FE measured-dead for trees); a smooth NN cannot synthesize them ⇒ handing them over
converts "learn a discontinuous rule" into "learn a near-linear map." Ranked: (1) **64-way
rule-combo TE** = cross-fitted P(class | stress×activity×sleep_quality) — the repo's
explicitly deferred-FOR-NN item; (2) **exact-value TE** of the 6 numerics (`train_te_num.py`
already emits `te_<col>_{0,1,2}`); (3) **ordinal INTEGER scalars** for stress/sleep_quality/
activity/smoking (monotone, not unordered embeddings); (4) 2-way driver crosses; (5) adv-shift
score as a feature. Backbone: RealMLP-TD (loss knobs + these inputs).

**ESCALATION (only if rungs move; ranked, below the fold)**: DCNv2 cross-network (cheap; label
is a literal bounded-degree conjunction) · PLE numeric-embedding tabular ResNet (smooth
interpolation over the coarse quantized grids, decorrelates from step functions) · SAM
(orthogonal flat-minima hedge — survives if the repaired trees already ate the mask lever) ·
DAE/SubTab masked-reconstruction pretraining · aux masked-driver head (cheap bolt-on to B) ·
TabPFN-v2 missing-region specialist (most-different prior, but capped by the measured 0.886
Bayes ceiling — calibration-diversity long shot).

**KILLED / not novel (do NOT build)**: pure re-weighting losses (focal / label-smoothing /
logit-adjustment / class-weights) = existing RealMLP knobs AND redundant with the decision
layer (weights-in-training ≡ prior-correction, substitutes; stacking cost −0.045 on s6e6) ·
distillation from the GBDT champion (anti-diversity — pulls the NN toward the base it must
differ from) · predict-missingness-as-aux-target (HARMFUL — hardwires the train-only coupling) ·
retrieval / kNN-augmented (`probe_linkage`: matching carries LESS info than the marginal) ·
SWA (≈ existing EMA knob) · exotic activations / self-normalizing nets / Lion / schedule-free
(bottleneck is data/mechanism-limited, not optimization-limited).

**GATE for every leg**: `diag_mlp_transfer.py` adv-weighted Δ>+0.001 AND test-like Δ>0 vs the
REPAIRED core; blend-overfit guard = split-half blend gate (≥6 seed swaps, mean holdout
Δ>+0.0005) before queuing, NOT raw Nelder-Mead OOF delta. LB-blind throughout. Seed-average
any survivor (5–10 seeds) as one stable leg.

**Honest EV (footnote, NOT headline)**: realistic best-case ensemble gain +0.0002–0.0005 on
repaired OOF, much of which may not transfer (no region has BOTH errors AND decorrelated-
transferable structure — complete-driver rows are solved, missing-driver rows are info-limited
/ correlated). BUT the crux re-gate is unrun and cheap ⇒ run rung 0/1 before concluding
anything (per [[feedback_premature_ceiling]]). **USER RULING (07-07 evening): NN day is
EXPLORATORY** — a sub-gate-but-genuinely-decorrelated NN can earn a final-#2 hedge slot; the
+0.001 bar gates *queuing a submission*, not *building*. Judge legs by decorrelation-on-
test-like-rows (diag fix-share + error-overlap), not blend OOF delta.

## ▶▶ PLAN for 2026-07-07 — OUTCOMES (see Day-5 section): #0 repair **PASS** (wired,
## champion_repaired built); #1 cost-matrix **FAIL** (split-half); #2 driver-posteriors
## **FAIL** (drivers unrecoverable); #3 HP tuning **FAIL** (hgbc canary +0.0004); #4
## NaN-semantics diversity NOT RUN (partly superseded — the repair already changed the NaN
## semantics globally; revisit only if a diversity gap shows); #6 TabPFN deferred (GPU
## booked by breadth); #7 breadth rerun WITH the repair (`_r` tags) — in flight.

User verdict on the old plan (breadth + falsifier + hygiene): intellectually lazy. Kill-audit
agrees — three Tier-C "kills" were ASSERTED, never measured: (a) HP tuning (zero Optuna trials
ever ran), (b) the 3×3 cost matrix ("subset of decision weights" is WRONG math — weights are
argmax(w_c·p_c), 2 effective params; the full matrix argmax_c Σ_j p_j·C[j,c] has ~5 and strictly
contains them), (c) "FE explored" = one probe with two transforms. The +0.0002 headroom oracle
bounds recombination of EXISTING bases only — new bases / new decision families sit outside it.
Standing gate for every candidate: adv-weighted Δ>+0.001 AND test-like Δ>0; LB-blind throughout.

0. **⚡ MISSINGNESS-MECHANISM REPAIR (new #1 — see Day-4-night section).** Quantify via
   test-mechanism re-masked OOF, then gate repairs R1/R2/R3. This targets ~13% of test rows
   with a measured train/test difference — the only lever with a demonstrated mechanism AND
   scale. All other candidates below should ALSO be scored on the re-masked OOF instrument.
1. **Cost-matrix decision layer (~2 h, first).** Fit the full 3×3 on blended OOF with
   CROSS-FITTED gain estimation (never trust in-sample NM on 5 params). Use
   `data_science_stuff.kaggle.decision.fit_cost_matrix`/`make_cost_matrix`. Mechanism: tilts
   the at-risk↔unhealthy boundary independently of at-risk↔fit — where 79% of flips live.
2. **Driver-posterior features (~half day, the FE centerpiece).** Aux models P(stress|x_obs),
   P(activity|x_obs), P(sleep_bucket|x_obs) trained on train+test observed rows (leak-free —
   no label involved; transductive = shift-adapted), posteriors fed to label models as
   features. Gives trees a "probably-high-stress" split exactly where the driver is NaN.
   Distinct from the marginalization probes: those replaced predictions; this adds inputs.
3. **HP tuning with the DEPLOYED objective (~half day, parallel).** Optuna, objective =
   decision-weighted balanced acc (optionally missing-region-weighted), NOT logloss. Order:
   hgbc first (~5 s/fold → 200+ trials/h on CPU), then LGBM, then XGB/Cat on GPU.
4. **Missingness-semantics diversity (~2–3 h).** Same GBDTs, different NaN treatments
   (median/model imputation; NaN-as-category for ordinal drivers) → decorrelates errors inside
   the missing region where all current bases share the same default-direction routing. Cheap
   add-on: cached adversarial score as a feature (shift-aware boundaries).
5. ~~id/row-order artifact probe~~ **DONE 07-06 night** (`eda_id.py`, Day-4-night-part-2
   section): id→target NULL, feature drift NULL — but found the test masks are blockwise in
   id (not exploitable; confirms uniform re-masking is the right #0 instrument).
6. **TabPFN-v2 missing-region specialist (~2–3 h, GPU-gated).** Subsample-ensembled TabPFN on
   the 180k missing-driver rows only — the one genuinely different prior never priced; the
   diversity oracle predates this base so does not bound it.
7. **Background only: 8-seed breadth run** (`run_breadth.sh`, resumable — CPU half + GPU half
   whenever the GPU is idle between items 3/4/6), then `combine_breadth.py` → final #2.
   Ceiling-falsifier (error-AUC probe) demoted to filler.

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
| 2026-07-07 | diag_repair r2a | R2a uniform re-mask of train, paired instrument: testmech 0.9467→0.9477 (**PASS**, mechanism damage fully recovered), plain +0.0001. Wired as `S6E7_REPAIR=1`. | 0.9477 (testmech) |
| 2026-07-07 | cost_matrix_probe | Full 3×3 Bayes matrix on champion blend, split-half gated: mean holdout **−0.00003** → measured-dead. | — |
| 2026-07-07 | tune_hgbc | Optuna 150 trials, deployed objective, repaired surface: best +0.0004 vs incumbent → tuning measured-dead for the GBDT family. | 0.9486 (repaired surf.) |
| 2026-07-07 | lgbm/xgb/cat `_r_s42` | Repaired champion legs (seed 42). Weighted 0.9478 / 0.9473 / 0.9479 on the repaired surface. | 0.9478 / 0.9473 / 0.9479 |
| 2026-07-07 | champion_repaired | v1b-recipe blend of the `_r_s42` legs (≈.33/.33/.35). 566 test flips vs champion, 2.8× NaN-enriched. **Submitted (user-directed) with `lgbm_r_s42`.** | **0.9484 (repaired surf.)** |
| 2026-07-07 | lgbm_dp_r_s42 | Driver-posterior features (+9 transductive aux posteriors): solo −0.0002, blend +0.0001 → **vetoed**; drivers measured ~unrecoverable (aux acc 0.48/0.56/0.72). | 0.9476 (repaired surf.) |
| 2026-07-07 | lgbm_freq_r_s42 | Frequency-encoding features (+7 pooled value-counts + combo count): solo +0.0001, blend +0.0001 → **vetoed**. | 0.9479 (repaired surf.) |
| 2026-07-07 | hgbc_r_s42..7777 | 8 repaired hgbc seeds, weighted 0.9481–0.9483 — near-zero seed variance; best single base on the repaired surface. | 0.9482 (repaired surf.) |
| 2026-07-07 | lgbm_te_r_s42 | Exact-value TE of 6 numerics (nested cross-fit, notebook-sourced): solo +0.0002, blend +0.0003 (day's best FE, still sub-gate) → **not queued**. | 0.9480 (repaired surf.) |
| 2026-07-07 | final2_breadth_r | **Repaired final #2**: 8-seed × 4-base breadth blend (32 fits; lgbm/hgbc/xgb/cat combined 0.9484/0.9484/0.9481/0.9481). | **0.9487 (repaired surf.)** |
| 2026-07-08 | gate fix | `diag_mlp_transfer.py` rewritten: repaired-breadth core, surface-keyed adv cache (repaired adv-AUC **0.6886** > plain 0.65), w=0.20 mixture gate + grid, error-overlap stats, surface guard. Sanity: core 0.9487 ✓, core member ≈ zero diversity ✓. | — |
| 2026-07-08 | mlp_r | Repaired MLP (3 seeds × 5 folds, flag-flip rerun). Gate: **VETO** (advwt +0.0000 / test-like −0.0001 at w=0.20). Signature = Day-2 exactly: fix-share 5.1%, 90.2% of fixes missing-driver, error-overlap 92.3% ≈ core member. | 0.9476 (repaired surf.) |
| 2026-07-08 | mlp_la_r | Repaired logit-adjusted MLP, same protocol. Gate left running at session end (`diag_mlp_la_r.txt`; expected VETO — identical OOF to mlp_r). NN avenue closed as measured on mlp_r; Rung 2 demoted. | 0.9476 (repaired surf.) |
| 2026-07-09 | mlp_la_r gate rerun | Day-6 loose end closed: **VETO** (advwt +0.0000 / test-like +0.0001 at w=0.20; fix-share 4.6%, 88.3% missing-driver, overlap 92.2%). Same signature as mlp_r. | — |
| 2026-07-09 | probe_tabpfn | TabPFN-v2 (10k ctx, 6 GB OK, 2.2 GiB peak) on 20k held-out missing-driver rows: 0.8827 w/ in-sample weights vs core 0.8970 same rows, recalls dominated → **priced DEAD**. | 0.8827 (region) |
| 2026-07-09/10 | mult sweep m000/m050/m100 | R2a overshoot probe (repair ~2× test NaN volume) + new volume-honest **TESTVOL** read. testvol 0.9476/0.9486/0.9484 → repair VALIDATED (+0.0008 vs none); mult 0.5 best on every surface but only +0.0002 (sub-gate) → **deployment stays mult 1.0, no rebuild**; `S6E7_REPAIR_MULT` wired for any future rebuild. | 0.9486 (testvol, m050) |
| 2026-07-10 | probe_error_auc | Ceiling falsifier: predict core OOF errors. conf 0.9538 / feat 0.9520 / both 0.9560 → **increment +0.0021 ≈ 0**; row-level limit corroborated. | — |
| 2026-07-10 | diag_repair r2a_te m050 | mult0.5 + exact-value TE on the paired chain: testvol **0.9488** (+0.0004 vs deployed m100) → residues stack; `_r2` rebuild gate PASS (runs overnight). | 0.9488 (testvol) |
| 2026-07-10 | extratrees/rf `_r_s42` | Zoo Z5 (`train_trees.py`): ET 0.9219 (veto; 97.7% of fixes missing-driver, overlap 50.9%); RF 0.9480 (veto; core-member signature). | 0.9219 / 0.9480 |
| 2026-07-10 | ftt_r_s42 | Zoo Z2 (`train_ftt.py`): FT-Transformer with ABSENT missing tokens (key_padding_mask; setenc-lite). Solo −0.0009; signature = mlp_r (77% miss-driver fixes, overlap 91.9%) → **VETO; Rung-2A structural prior measured-dead in cheap form**. | 0.9478 |
| 2026-07-10 | **realmlp `_r_s42/s7/s123`** | **Zoo Z1 (`train_realmlp_td.py`): RealMLP-TD + TE input recipe. 0.9492 weighted EVERY seed — first solo leg above the core (+0.0005, uniform across regions incl. test-like). Formal w=0.2 gate veto (+0.0003) but fixed-w mixtures monotone-improve to w=0.5.** | **0.9492 (solo, repaired surf.)** |
| 2026-07-10 | **final3_realmlp** | **NM blend {4 GBDT `_r_breadth`} + `realmlp_r_breadth` (3 seeds): realmlp w=0.845, OOF 0.9494. Split-half gate 12/12 positive (mean +0.00042); complete-in-4 +0.0006 paired. ⇒ final #2 (superseded same evening by metablend_r).** | **0.9494 (repaired surf.)** |
| 2026-07-10 | lrstack_r / lrstack_pc_r | s6e6 LR-on-logits stacker ported (`build_lr_stack.py`, honest 25-fit cross-fit + decision weights). Variants identical; honest OOF ties final3's optimistic OOF ⇒ genuinely ahead. Combiner gate: LR−NM +0.00032 (10/12). | 0.9494 (honest, repaired surf.) |
| 2026-07-10 | run_caruana | Greedy selection over 12 repaired legs: holdout 0.94938 ≈ LR tie; picks realmlp 62% (third confirmation); vetoed legs earn only noise-level seats. | 0.94938 (holdout argmax) |
| 2026-07-10 | **metablend_r** | **avg(final3_realmlp, lrstack_r) + decision weights — combiner gate 11/12 positive vs NM (+0.00032, lowest sd); miss4 best (0.9443). ⇒ NEW FINAL #2 (s6e6 best-LB shape).** | **0.9494 (repaired surf.)** |
| 2026-07-10 | combiner tournament | 8 arms × 12 split-half cells: lr6 wins (+0.00040 vs NM, 11/12; mlp_la_r earns its seat per-class); gbdtmeta/nnmeta positive but lose to LR; lr6 vs metablend_r +0.00008 = below displacement gate → final #2 unchanged. | lrstack6_r 0.9495 (honest) |
| 2026-07-10 | dann_r_s42 | Zoo Z4 (grad-reversal + dual-mask consistency, semi-supervised over test): solo 0.9472; **VETO** — 95.2% of fixes missing-driver, overlap 91.7% = the universal NN signature. **Rung 2 closed by measurement (4 mechanisms).** | 0.9472 (repaired surf.) |
| 2026-07-10 | _r2 lineage endgame | Full mult-0.5 rebuild priced at deployed level via intersection reads: ensemble_r2_gbdt 0.9505 vs core 0.9506 (wash); final5r2_r 0.9512 = tie with metablend_r. Own-surface +0.0005 = inflation. **m050 lineage dead; do not chase.** | final5r2_r 0.9499 own / 0.9512 inter. |
| 2026-07-10 | tabm_r_s42 | Zoo Z3 (overnight retry, ~24 min/fold): argmax 0.8774, weighted **0.9411**. NOT an architecture verdict — pytabkit TabM has NO class-weight/loss hook (s6e6 lesson repeating): plain CE on 86/8/6 majority-walls the minorities and decision weights can't fully rescue a train-time miscalibration. A "fixed" TabM ≈ what realmlp already is (deep ensemble + balanced softmax), so not worth rehabilitating. **Zoo scoreboard complete: 6 families priced, one survivor (realmlp).** | 0.9411 (repaired surf.) |
| 2026-07-10 | lgbm_rule | Exact-rule three-valued deduction features (`rule_features.py`, from discussion/717222's proven generation tree; 89.1% of rows logically determined incl. 63.5% of missing-driver rows): **flat for trees** (−0.0002 vs paired lgbm_r_s42 0.9478) — they had it statistically. Realmlp consumer stays live (tomorrow #2). | 0.9476 (repaired surf.) |

**LB**: lgbm 0.94886, xgboost 0.94894, **ensemble_v1b 0.94970 (best)**. The blend lifts once
models are pre-corrected to their deployed surface. Next lever for a bigger jump is a non-GBDT base
(the 3 GBDTs still only fix ~5–7% of each other's errors).

### Fold-score note
`cv_scores.csv` fold_* columns are plain-**argmax** balanced accuracy (fold variance), while the
`oof_balanced_acc` headline is the **decision-weighted** score — do not compare a fold column to the
headline directly.
