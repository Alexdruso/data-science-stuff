# Day-10 idea backlog — three fresh-persona brainstorms (2026-07-11 night)
# Personas: TRAINING-PROCESS ENGINEER / GENERATOR FORENSICS / LB ANTHROPOLOGIST.
# Synthesized ranking at the bottom. Full agent outputs condensed; kill-list-aware.

## STRATEGIC HEADLINE (anthropologist, web-verified with sources)
- Public LB top (0.9510-0.9517, 86 teams ≥0.9510) is ~entirely LB GAMING: majority votes
  over shared public submissions + per-row LB-probed label overrides (beicicc notebook:
  122 hand-coded test-id flips; anhadmahajan06 pseudo-ground-truth from LB reads;
  hikari30 top-8 majority). These do NOT transfer to private.
- Best documented HONEST CVs in the field: 0.95067 (Kawamata LogReg stacker), 0.95063
  (FT-Transformer+TE), 0.95063 (yunsuxiaozi RealMLP) — ALL BELOW our lrstack_u2/u3
  0.9508. We are at/ahead of the honest frontier.
- ⇒ PROTOCOL: finals by CV only; treat public reads ≥0.9508 as contaminated; expected
  winner = TE-rich multi-family stack + prior correction + private draw = what we hold.
  Variance protection (seeds) beats mean-chasing from here.
- Field-verified kills to respect: BA as early-stopping/selection objective TOXIC
  (select on balanced logloss); global/budgeted minority-count promotion HURTS
  (LB-verified −0.0009); missingness MCAR confirmed (flags redundant with NaN-TE-level);
  original-50k-as-training-rows dead (matches our kill).
- Nuggets: geometric/log-space member averaging > arithmetic (+0.00012, amerhu);
  per-value TE cannot be screened at small n (−0.0017 at 70k vs +0.0012 at 690k) — why
  everyone else missed it.

## TRAINING-PROCESS ENGINEER (ranked)
1. **Provable-noise curation** (P .30, the day's headline): ~4,800 train rows where
   observed drivers CONTRADICT the label = known generator noise. Three arms:
   relabel-to-rule / drop / soft-label 0.992; + variant excluding them from TE fitting
   only (isolates TE-contamination channel — noise poisons the exact-value TE tables
   that just paid, and mislabeled minority rows carry 15x metric weight). Probe: 3
   paired lgbm_te3 arms, 1h.
2. **Regime-conditioned meta** (P .22, pure OOF math): interact per-(model,class) stack
   logits with the 8 driver-missingness-pattern one-hots + sweep C in {0.03..3}. Not
   the dead raw-features-in-meta (no feature values) nor dead per-region decision
   weights (upstream, on probabilities, with LR shrinkage). Probe: 30 min, no training.
3. **Decision-weight generalization audit** (P .20): LOFO cross-fitted NM weights vs
   full-OOF fit — the +0.010 layer has never had its optimism measured; 5% optimism =
   0.0005. Probe: 30 min OOF math. Deploy LOFO-averaged/shrunk weights if gap >0.0005.
4. ~~Balanced-acc early stopping~~ — DEMOTED: field-verified toxic (anthropologist #4);
   if touched at all, use balanced LOGLOSS as the eval metric instead.
5. **Per-class specialist binaries as meta columns** (P .14): binary lgbm_te3
   fit-vs-rest / unh-vs-rest logits appended to the meta design (adds inputs, not the
   dead cascade decision structure; binary-target TE = different statistic). 45 min.
6. **RealMLP epoch/snapshot averaging** (P .12): best epoch 4-6 = steep-trajectory
   point estimate; window-average softmax = free variance on the stack's NN column.
7. Missingness TTA (P .07): inference-only filler.

## GENERATOR FORENSICS (ranked)
1. **Pairwise co-occurrence TE** (P .15): the generator ADDED ~0.10 cross-feature
   dependence (absent in original, shared train/test); univariate te13 can't represent
   P(y|v_i,v_j) and the 33-leaf regularization prevents trees from building it. Rank 78
   pairs by |corr_synth − corr_orig|, top ~15 pair-string EB-TEs (+45 cols), paired
   hgbc_te3 seed; watch the miss-driver slice (observed-pair joint proxy for a masked
   driver — NOT the dead per-column aux posteriors).
2. **Original-world naive-Bayes prior columns** (P .10): NB is correctly specified on
   the ORIGINAL (|corr| 0.003, unlike synthetic); 3 log-posterior columns from
   per-value P_orig(v|y) as features (clean pre-generator prior where TE counts are
   distorted; freq-corr down to 0.58). Distinct from dead row-augmentation + dead
   synthetic-world analytic posterior.
3. **Impossible-pattern coverage** (P .08): 3,703 test rows have joint missingness
   patterns with zero train support; inject pattern-matched remasked copies (targeted,
   not the dead uniform repair) or marginalize orphans over nearest seen patterns.
4. **Frequency-distortion features** (P .07): 10-min decisive pre-check — are per-value
   frequency distortions class-conditional (CTGAN conditional-vector signature)? If
   yes: per-column log-count + freq-ratio-to-original + row distortion sum + copula
   typicality.
5. **Rare-value purity** (P .06): if the resampling generator memorizes, rare values are
   PURER than EB predicts; bucket-by-count purity curve; if monotone, add a low-smooth
   TE block.
6. **Row-memorization exact-match transfer** (P .04 but 15-MIN PROBE): hash-join test to
   original on all 13 string columns; if matches >> permuted null AND train-match label
   agreement >0.992, overwrite exact-match predictions post-hoc.
7. Empirical coarse-key cell posterior (P .05): full-key EB-TE (no independence
   assumption — the dead one was analytic); only if key coverage n>=30 for >80% of test.

## LB ANTHROPOLOGIST (ranked, actionable)
1. **FT-Transformer + per-value TE as a 4th stack family** (P .25): independently
   verified CV 0.95063 (Kawamata, pip masamlp, te_mode="direct", 16 epochs @ bs 4096);
   our FTT kill predates the TE-consumer era. Probe: 1 fold ~40 min GPU.
2. **Blend hygiene** (P .15): geometric member averaging + balanced-logloss (not BA)
   selection objective inside the combiner; recheck the tournament under both.
3. TabPFN-3 subsample base (P .10-.15): new foundation-model family; our TabPFN-v2 kill
   was region-restricted. Cheap GPU probe.
4. **Seed-bagging for private-draw variance** (P(rank protection) high): s6e6 precedent
   — private winner was public #343. Extend champion-stack seeds; select by CV.

## SYNTHESIZED DAY-10 ORDER
- **Before coffee (pure math, no training)**: TE-eng #2 (regime meta) + TE-eng #3
  (decision-weight audit) + GF #6 (15-min memorization hash-join) + GF #4's 10-min
  class-conditional frequency check.
- **Morning training axis**: TE-eng #1 noise-curation 3-arm (CPU) ∥ Anthro #1 FTT+TE
  probe (GPU) ∥ submit yesterday's 4 queued CSVs (realmlp_te13_u_s42,
  hgbc_te3_u_breadth, lgbm_te3_u_s42, lrstack_u3_r).
- **Afternoon**: GF #1 pairwise TE (if morning slots free) + winners' consumers +
  combiner hygiene (Anthro #2) + stack rebuild with any new leg.
- **Standing strategy**: finals by CV; public ≥0.9508 contaminated by probing; variance
  protection via seeds; do NOT build LB-override/blend-of-public machinery.
