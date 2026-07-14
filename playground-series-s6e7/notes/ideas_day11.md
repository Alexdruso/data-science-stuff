# Day-11 idea backlog — NEW-channel brainstorm under the 3-dim label-channel model (2026-07-13)

Framing persona: creative GM screening the FULL standard tabular playbook against ONE proven
constraint. This is exploration phase (weeks of runway); we do NOT curate labels (flips are
learnable generator structure); we work LB-blind (discard <0.001 expected-effect ideas).

## The screen (apply to every idea)

A generator-forensics analysis PROVED the label channel is exactly **3-dimensional**: conditional
on (stress_level × physical_activity_level × EXACT sleep_duration value), P(label-flip) is
statistically independent of everything else — including a 0.908-AUC original-vs-synthetic
discriminator (p=0.40). The label = a near-deterministic depth-4 rule on (sleep 6/7 thresholds ×
stress × activity) + a 0.78% cell-clustered noise tail (CTGAN-class generator). The all-13-column
string TE recipe ("te13") already SATURATES this channel, confirmed by five downstream reads
(pair-TE, kNN-on-trees, kNN-on-NN, unweighted-solo, seat tests — all Day-10).

**Three verdict buckets:**
- `SAME-3-DIM-VEHICLE` — a different way to feed the same (stress × activity × exact-sleep)
  information (new architecture / new feature / new TE variant / new imbalance trick). KILL.
- `ALREADY-MEASURED-DEAD` — cite the specific kill.
- `ORTHOGONAL-CHANNEL` — injects information the triple does NOT mediate. Only two sub-types can
  qualify: **(A) genuinely-new external signal**, or **(B) an objective/target/decision
  formulation not mediated by the triple**. These survive the screen and go to the shortlist —
  but still get an honest prior.

Crucial reclassification (per the premature-ceiling lesson): "feature channel saturated" is NOT
"everything closed." The feature-vehicle bucket is dead; buckets A and B are structurally live and
are where the remaining EV sits. The shortlist below is thin but non-empty by construction.

---

## The full playbook, screened (18 directions on the record)

### Feature-engineering family
1. **More TE variants** (pair/triple/exact-value/kNN-label-dist/NB-prior/freq-distortion).
   Mechanism: richer posterior encodings of the same columns. → `SAME-3-DIM-VEHICLE` /
   `ALREADY-MEASURED-DEAD` (Day-5 6-family FE sweep; Day-10 pair-TE, kNN, NB, freq-distortion all
   FLAT; te13 saturates). Prior none, cost n/a.
2. **Generic FE** (ratios, sleep-debt, BMI-bands, interactions, missingness-count). → `SAME` /
   `DEAD` (Tier-C, ablation −0.0005). Prior none.
3. **Missingness indicators / pattern features.** → `DEAD` (MCAR in test; indicators add 0.0000;
   train-only trigger recovery is test-invalid). Prior none.
4. **Feature selection / drop water_intake.** → `DEAD` (water-drop flip fragility, Day-2; noise
   column still tie-breaks splits). Prior none.
5. **Driver-posterior / soft-completion features** (impute the masked driver, feed P(driver|x)).
   Mechanism: recover the missing triple component. → `ALREADY-MEASURED-DEAD` (Day-5 lgbm_dp
   −0.0002; drivers ~unrecoverable, stress-aux 0.48 vs 0.43 majority; test masks MCAR ⇒ x_obs
   carries no info about the masked driver). This is the crux of the 86%-MCAR error floor. Prior
   none.

### Model-family family
6. **New GBDT configs / HP tuning** (Optuna over LGBM/XGB/Cat/HGBC). → `SAME` (vehicle) /
   `DEAD` (Day-5 hgbc canary +0.0004, TPE converged flat). Prior none.
7. **New NN architectures** (MLP / logit-adjust / FTT / DANN / TabM / RealMLP / distill). →
   `SAME` + `DEAD` (Day-6→10: every NN concentrates diversity in the info-deleted missing-driver
   region, none transfers; RealMLP/te13 already the strong base). Prior none as *new info*.
8. **Foundation models** (TabPFN-v2/2.5/3 as base or missing-region specialist). → `SAME` /
   `DEAD` (Day-7 priced 0.8827 in-region vs core 0.897; capped by the 0.886 Bayes ceiling). Prior
   low, cost med (TabPFN-3 subsample is the only literally-unrun variant; still a triple vehicle).
9. **Rule-based / monotonic / EBM / RuleFit / linear-tree.** → `SAME` / `DEAD` (Tier-C; GBDT
   already beats the hand-rule; rule-deduction features FLAT Day-8/9). Prior none.

### Imbalance / objective family
10. **Reweighting losses** (focal / label-smoothing / logit-adjust / class-weight / effective-num).
    → `SAME` (substitutes for the decision layer; prior-correction ≡ weights, stacking both costs
    −0.045) / `DEAD`. Prior none.
11. **Cost matrix (full 3×3).** → `ALREADY-MEASURED-DEAD` (Day-5 split-half −0.00003; per-class
    weights saturate the decision layer). Prior none.
12. **Chained binary cascade / hierarchical stages.** Mechanism: binary objective changes what the
    trees LEARN on the live boundary. → `ALREADY-MEASURED-DEAD` (Day-9: +0.0004 solo but 94.3%
    error-overlap, negative LR-seat — *better leg, identical errors*). This is the load-bearing
    precedent: a boundary-targeted OBJECTIVE reshuffled the trained boundary yet produced the same
    errors, because they live in the MCAR region no objective can fix. Prior low for the whole
    objective sub-channel.
13. **Differentiable balanced-accuracy surrogate** (soft confusion matrix / learned margins).
    Mechanism: optimize the metric directly instead of via the decision layer. → `ORTHOGONAL-
    CHANNEL (B)`, **literally unmeasured** (Day-9 alchemist #7, never run). Screen honesty: the
    features are triple-dominated, so any objective is partly triple-mediated, and cascade shows a
    boundary objective yields identical errors ⇒ prior LOW, likely same-outcome. Cost low
    (one realmlp/lgbm loss swap). SHORTLIST (tail).

### Data / semi-supervised family
14. **External data** (the 50k seed, or a *new* real-world student-health set). → `ALREADY-
    MEASURED-DEAD` (Day-3 augmentation −0.0007 teaches the solved regime; Day-4 linkage at chance;
    a new real set won't match the synthetic RULE and can't linkage-join). Prior none.
15. **Pseudo-labeling / self-training / label propagation.** → `DEAD` (mechanism = augmenting with
    already-solved rule-determined rows; Tier-C). Prior none.
16. **Transductive / semi-supervised use of the 295k test rows** (true mask mechanism). →
    `ALREADY-MEASURED-DEAD` (Day-9 distillation on real test = twin exactly; P(y|x) shift absent).
    Prior none — EXCEPT as an objective anchor, see #18.
17. **Synthetic pretraining** (fit CTGAN/generator, sample 50-100M, pretrain, fine-tune). →
    `SAME-3-DIM-VEHICLE` (the generator replays P(y|triple)+noise; adv-AUC gate would likely fail;
    Day-9 deprioritized as "same channel, 10× cost"). Prior low, cost high.

### Objective / decision / meta family
18. **Error-placement (anti-correlation) penalty vs frozen core, computed ONLY on complete ∩
    test-like rows** (Day-9 alchemist #3). Mechanism: instead of a generic new loss, make
    *transferable decorrelation* the explicit training objective — penalize agreement with the
    frozen core exactly where information EXISTS (complete-driver ∩ test-like), so the model is
    forbidden from parking its diversity in the MCAR region where every prior NN dumped it. →
    `ORTHOGONAL-CHANNEL (B)`, **confirmed never run** (not in the experiments log — DANN did
    grad-reversal + dual-mask consistency, a *different* objective; this penalizes the measured
    failure signature directly). Screen honesty: still fights for scraps in a region where the
    core is already near-Bayes, and cascade warns objective levers reproduce errors ⇒ prior LOW —
    but it is the ONE objective aimed at the *measured* failure mode, not a generic reshuffle.
    Cost med (one GPU training script, semi-supervised loader already exists from DANN).

### Combiner / stacking / calibration family
19. **New meta-learner / regime-conditioned meta / Caruana / metablend.** → `DEAD` (combiner axis
    measured saturated, 5+ consecutive LR-seat vetoes; regime-meta 3 independent vetoes). Prior
    none.
20. **Calibration / prior-correction / metric-aware post-proc.** → `DEAD` (decision layer
    saturates; LOFO audit shows the +0.010 layer generalizes, weights ≈[1,1,1]; Jensen
    random-denominator correction ≈6e-5 negligible). Prior none.

### Variance / finals family (orthogonal by construction — not an information channel)
21. **Seed bagging / breadth for private-draw variance.** → `ORTHOGONAL (not info)` but `SEEDS
    SPENT` (per-seed sd ~1e-4 vs 6e-4 private-split sd; Day-10 decision-theorist). Prior: variance
    protection real but exhausted at the margin. Cost free-ish.
22. **E[max]-of-two-submissions portfolio + finals selection.** → `ORTHOGONAL (not info)`; this is
    where the actual residual EV lives, and it is heavily worked (probe_emax flipping DEAD;
    probe_finals_pairs; δ-bracket). Standing item, not new. Prior: the real game, but no new lever.
23. **Public-LB probing** (`lbprobe_*`). → `ORTHOGONAL external` but public∩private=∅ ⇒ ZERO
    private/finals EV; quarantined for-fun only. Prior none for the real goal.

### External-information activity (process, not experiment)
24. **Daily forum / notebook sweep for a new recipe or signal.** → `ORTHOGONAL-CHANNEL (A)`, the
    empirically **highest-yield** route: the campaign's single largest gain (+0.0014) came from a
    forum post (Susol te13), vs +0.0009 from six days of internal probing. Standing item flagged
    #1 by the Day-10 decision theorist. Prior: highest-EV overall; cost ~10 min/day.

---

## Shortlist — survivors that pass the orthogonal-information screen

The feature/information channel is measured-saturated; the honest survivors are all in buckets A
(external) and B (objective/decision), and their priors are low. This is a thin shortlist, stated
as such — but it is not empty, and it is not a closure argument.

**S1. Error-placement (anti-correlation) penalty on complete ∩ test-like rows** — the single
highest-value genuinely-NEW experiment. It is the only objective-channel lever that (a) is
literally unrun, and (b) attacks the *measured* failure signature (every NN dumps diversity in the
MCAR region) instead of being a generic reshuffle. Cheapest decisive experiment: train one
realmlp/te13 leg with an added penalty `+λ·corr(pred, frozen-core-pred)` evaluated ONLY on the
complete-driver ∩ test-like subset (semi-supervised loader exists from DANN); sweep λ∈{0.1,0.3,1};
success = diag fix-share rises AND `%fixes-missing-driver` FALLS below the universal ~90% for the
first time, then a 12-cell LR-seat test. Kill if the signature stays ≥90% missing-driver (it will
have reproduced the cascade/DANN outcome). Prior LOW; cost ~½ day GPU.

**S2. Differentiable balanced-accuracy surrogate leg** — the last unmeasured objective slot;
weaker than S1 because it is a generic new loss, not aimed at the failure mode. Cheapest decisive
experiment: swap the balanced-softmax for a soft-confusion-matrix BA surrogate in one lgbm-te3 /
realmlp leg, paired vs its twin; gate = diag signature + LR-seat. Prior LOW (cascade precedent
says likely identical errors); cost low. Run only if S1 moves.

**S3. Continue the daily external-info sweep** — NOT new, but the highest-EV *activity* and the
only route that has ever actually injected orthogonal information. Keep it standing.

**S4. (Adjacent, not information) sharpen the finals/E[max] instrument** — the real residual EV is
the two-submission variance game, already well-worked; no new lever, listed for completeness so it
is not mistaken for closed.

## The single highest-value genuinely-new direction (even at low prior)

**S1 — the error-placement penalty.** Rationale: of the two orthogonal buckets, external-info (A)
has no *new* experiment (only the standing sweep), so the one genuinely-new experimental direction
must come from bucket B. Among B, S1 dominates S2 because it targets the campaign's *measured*
failure signature directly — "NN diversity concentrates where information was deleted" — rather
than reshuffling the boundary generically. Its prior is honestly low (cascade proved boundary
objectives can reproduce errors; the target region may simply be near-Bayes), but it is the most
mechanistically-motivated unmeasured shot on the board, and its diag instrument gives a clean,
cheap, falsifiable read: does *any* objective produce transferable (non-MCAR-region) decorrelation
for the first time in the campaign, or is the placement of the residual error itself a hard
invariant? Either answer is decision-relevant and closes an open question rather than asserting
one.

## Honest bottom line

The feature/information channel is saturated by measurement, not assertion. Live directions remain
only in the two orthogonal-by-construction buckets, and their priors are low. The highest-EV
*activity* is the external-info sweep (S3); the highest-value genuinely-NEW *experiment* is the
error-placement penalty (S1). Everything else in the 18-item playbook is a 3-dim vehicle or an
already-cited kill.

---

## RESULT (2026-07-13) — S1 built and measured: objective channel closed at its most-motivated point

Pre-check `probe_s1_oracle.py`: perfect selector over {core + 24 diverse bases} on
complete∩test-like = **+0.00176** (≈+0.0004 overall UB) → borderline target → user
approved the build. `train_s1.py` (DANN backbone, penalize core-agreement on
complete∩test-like), λ-sweep:

| λ | wbacc | fix-share | %fix miss-drv | err-overlap | fixes/breaks | LR seat |
|---|---|---|---|---|---|---|
| 0.1 | 0.9486 | 5.9% | 82.0% | 83.0% | 2414/5439 | −0.00000 (5/12) |
| 0.3 | 0.9398 | 11.8% | 91.4% | 63.4% | 4809/15922 | +0.00001 (3/12) |
| 1.0 | 0.9025 | 55.4% | 95.2% | 30.2% | 22562/19406 | +0.00002 (7/12) |

The objective **worked mechanically** — error-overlap collapsed to 30% (most
decorrelated leg of the campaign; λ=0.1 bent the missing-driver invariant to 82%) —
but produced **zero transferable value**: net-negative solo at every λ, no test-like
enrichment, **no LR seat** (6th consecutive veto). Confirms the decomposition "no
region has BOTH errors AND decorrelated-transferable structure" at its sharpest. ⇒ S1
is inventory; **bucket B (objective/decision) is measured-closed at its most-motivated
point**. Remaining live EV: bucket A (external sweep — S3, standing daily) and S2
(differentiable BA surrogate — weaker, now lower-prior given S1's outcome). Not a
ceiling: a pre-registered experiment ran and answered; exploration continues.

## RESULT (2026-07-13) — TabPFN-3 (bucket-A-adjacent untried family) measured out

The one untried model family (user-requested). tabpfn 8.1.0, isolated venv, license-
gated (TABPFN_TOKEN + accept ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3; no
sklearn conflict). Runs on the 6 GB card (0.87 GiB peak, ~12 min/fold). Probe passed the
gate (in-league + decorrelated) so a full 5-fold OOF was built (`train_tabpfn3.py`):
full-OOF dec-weighted **0.9482** (−0.0026 vs core), err-overlap 83.5%, %fix-miss-drv
**72.9%** (best decorrelation profile of the campaign) — but net-negative solo, no test-
like enrichment, and **no LR seat (+0.00000, 6/12; 7th veto)**. The combiner axis is
saturated even against a foundation-model prior. ⇒ last untried model family is measured
out. Only bucket A (external sweep, S3) remains live.

## RESULT (2026-07-13 pm) — 3-Haiku fresh-crack brainstorm: all held

User asked to "think of something" via a couple of Haiku agents. Three agents each
attacked a specific crack in the saturation thesis:
- **Encoding**: the one genuinely-unbuilt feature = exact 3-way cross-fitted
  P(y|stress×activity×EXACT-sleep) (`train_hgbc_te3_tri.py`). BUILT + MEASURED:
  **0.9501 vs hgbc_te3 0.9504 = −0.0003, flat**. Superset of the flat exact-pairs adds
  nothing over univariate te13. Encoding channel closed BY MEASUREMENT.
- **Decision/metric**: closed by PROOF — BA linear in per-row assignments ⇒ E[BA]-optimum
  IS argmax(w·p) with w∝1/E[n_c]; no cross-row coupling beyond fixed denominators.
- **Generating-process**: every angle collapses to a measured kill.

No new lever survived. Disciplined negative (attacked cracks, measured the testable one).

## RESULT (2026-07-13 eve) — user challenge: invent it ourselves, don't defer to the forum

User (verbatim): "if someone is inventive enough to find an improvement and publish it
in discussions, doesn't this mean you could also come up with something? I find it silly
that you think discussions are better than just trying ideas ourselves." → dropped the
"forum is the highest-EV lane" framing (the `feedback_premature_ceiling` pattern). Ran
two genuinely-orthogonal in-house probes that attack channels the whole zoo is BLIND to
(the 3-dim thesis was built from an AGGREGATE stat scan; aggregate tests cannot see
row-level or distributional structure):

- **Original-dataset hunt** (`data/original/`, web-searched, `kaggle datasets download`):
  3 candidates. `ziya07/student-health-data` (1k rows) and `chik0di/health-lifestyle`
  (100k) — wrong schema. `evan65549/health-and-fitness-dataset` = **687,701 rows ≈ our
  690,088 train** (strong "generator base" tell) BUT label space is disease
  {Hypertension/Diabetes/Asthma} + numeric stress — no clean-merge ground truth for our
  {at-risk/fit/unhealthy}. Synthetic features carry no key back to source rows anyway
  (advisor). **No free labels. Measured negative, not deferral.**

- **Row-level memorization** (`probe_nn_memorization.py`, NEW): raw kNN base — never in
  the zoo. (1) test→train nearest-neighbour distances: min 0.242, **frac<0.10 = 0.0000,
  frac<0.01 = 0.0000** (train→train min 0.176) → generator emits NO near-duplicates.
  (2) kNN OOF BA 0.734 (vs 0.951) and error-overlap 0.139 looked decorrelated, but a
  direct blend check is **negative at every weight** (Δ ≤ 0) → the low overlap was noise
  placement, not orthogonal truth. **Memorization channel closed by measurement.**

- **Covariate-shift importance-weighting** (`probe_covariate_shift.py`, NEW): the one
  MEASURED-but-unused signal (AV AUC 0.654). Importance-weight te13-HGBC toward the test
  distribution (density-ratio weights, max 11.8 but frac>1.5 only 1%); score plain +
  AV-weighted OOF BA. **Weighted vs unweighted: plain 0.9503 vs 0.9502, AV-weighted Δ =
  −0.0000 → FLAT.** The shift is PURE COVARIATE (P(x) moves, P(y|x) fixed) ⇒ Bayes
  predictor unchanged, exactly as theory predicts absent concept shift. Closed.

Orthogonal-axis map after the user challenge: features→label (saturated by zoo),
memorization (closed — no near-dupes, kNN blend-negative), train/test shift (closed —
pure covariate, weighting a no-op). Three axes I invented and tested MYSELF, three clean
measured negatives — the honest answer to "couldn't you invent something": I did, and
each orthogonal hypothesis is falsified by our own measurement. This is NOT a global-
ceiling declaration (that stays forbidden); it is the current evidence map. The deployed
posterior sits at/near the Bayes-optimal for this generator on every channel we can name.
