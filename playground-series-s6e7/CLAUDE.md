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

**LB**: lgbm 0.94886, xgboost 0.94894, **ensemble_v1b 0.94970 (best)**. The blend lifts once
models are pre-corrected to their deployed surface. Next lever for a bigger jump is a non-GBDT base
(the 3 GBDTs still only fix ~5–7% of each other's errors).

### Fold-score note
`cv_scores.csv` fold_* columns are plain-**argmax** balanced accuracy (fold variance), while the
`oof_balanced_acc` headline is the **decision-weighted** score — do not compare a fold column to the
headline directly.
