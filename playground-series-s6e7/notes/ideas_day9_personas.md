# Day-9 idea backlog — three Kaggler-persona brainstorms (2026-07-10 night)
# Full agent outputs preserved verbatim below; see CLAUDE.md tomorrow-queue for the committed items.
# Personas: LEAK HUNTER / DL ALCHEMIST / PROBABILITY PURIST.

## LEAK HUNTER (ranked; unifying read: train/test = different pipeline runs, id-ordered, stateful RNG)
1. R4 repair = DROP trigger-masked rows (~13%) + remask survivors at exact test rates — removes the coupling at source (R2a only dilutes; explains adv-AUC 0.6886). Probe: diag_repair chain, paired TESTVOL. P≈15%.
2. Test-head segment check: are pre-row-40,467 blocks still in TRAIN trigger mode? Per-segment mask×trigger contingency (15 min). If yes → segment-routed submission. P≈7%, big payoff.
3. Gender step-vs-drift at train/test boundary (5k-id bins, look for step at 690,088) — decides "two generator runs" vs within-split drift. 10 min. P(actionable)≈8%.
4. MNAR sub-selection inside trigger populations: AUC(water-mask | other cols) within gender=female; same in test active blocks. 20 min. P≈10% signal / 4% converts.
5. Label-noise burstiness in id: rule-disagreement (from rule_features, 89.1% determined) per id-bin + change-point test → if bursty, drop/downweight noisy segments. 30 min. P≈8%; real lever if fires.
6. Generator fingerprinting: (a) value-vocabulary containment vs original 50k; (b) within-class correlation shrinkage (⇒ conditional-marginal sampler ⇒ ceiling provably exact); (c) fit SDV/CTGAN to match z-std 0.97. Endgame: analytic generator posterior in missing region. 30 min for a+b. P(identify)≈15%, P(score)≈5%.
7. Shared-RNG mask nesting in test blocks (co-missingness vs independence, 10 min) → fix _uniform_remask pattern co-occurrence. P≈10%/3%.
8. FFT batch-size scan over id streams (rule-disagreement, masks, gender) for period-B generator artifacts. 15 min. P≈3% lottery.

## DL ALCHEMIST (ranked; organizing read: "hand the net computed quantities" is the only channel that ever paid; end every probe at the combiner gate, not the solo gate)
1. Generator-exact synthetic pretraining: known rule + noise level + copula of proxies + exact test mask mechanism → sample 50-100M rows, pretrain realmlp-recipe, fine-tune on real. Probe first: synthetic-vs-real adv-AUC ≲0.55 gate. P≈12%.
2. Transductive distillation from metablend on REAL TEST rows (only data with the true mask mechanism), then label fine-tune. NOT the anti-diversity kill (channel = surface, not diversity). P≈10%.
3. Anti-correlation penalty vs frozen core computed ONLY on complete∩test-like rows — makes decorrelation PLACEMENT the objective (first attack on the signature itself). Watch fix-share region split. P≈8%.
4. Soft-completion posteriors: compose driver-posterior aux models with the exact-rule/TE table (CORRECTED triple) into per-class marginalized posterior as realmlp input. P≈8%.
5. Pairwise ranking loss on at-risk-vs-minority (rank ≠ reweighting; changes ordering across the live boundary). Compare signature with cascade. P≈7%.
6. Per-row temperature/evidential head conditioned on missingness pattern → manufacture calibration-diversity for the LR meta (lr7 arm in the tournament; zero submission risk). P≈6%.
7. Differentiable balanced-accuracy surrogate (soft confusion, learned margins) — the one objective-engineering slot never measured. P≈5%.
8. SAM on realmlp (flat-minima transfer hedge; escalation condition "a rung moved" now met). Keep only if test-like Δ moves. P≈4%.

## PROBABILITY PURIST (ranked; jurisdiction: the max-of-two-submissions game + variance mathematics)
1. E[max] portfolio: E[max(S1,S2)] ≈ μ + 0.40·σ_Δ for equal means — flip K≈4k indifference-band rows toward minorities in final #2 (220:1 variance lever from balanced-acc denominators; zero-mean perturbation has capped downside). Probe: private-split Monte Carlo with posterior-uncertainty; sweep K. P≈20-25%. THE "theorem not hope" item.
2. Paired-delta calculus: closed-form sd of public/private deltas from two CSVs + posterior → price every LB read the user already made; punchline: public and private deltas of near-clones are nearly INDEPENDENT (disjoint rows) — public tiebreaks between near-ties carry ~no private info. P≈30% it changes/hardens a finals call.
3. Empirical-Bayes shrinkage of blend posterior toward exact-rule CELL posteriors (λ cross-fit; snap on logically-forced cells) — James-Stein on the decision surface, variance-only. P≈10%.
4. Analytic posterior arbitrage on the 3,703 train-impossible-pattern rows (exact rule marginalized + noise matrix vs extrapolating model). Probe on remasked OOF emulation, labels in hand, one afternoon. P≈8%.
5. Antithetic remask streams for every future gate (~√2 sharper instruments; the 0.001 gate line flickers on a noisy ruler). P≈70% it sharpens evidence; indirect value.
6. Rank-convexity sizing: user eyeballs LB histogram density around 0.9495-0.9500 → converts idea-1's σ_Δ into E[rank] and sizes K rationally.
7. Jensen micro-correction for random recall denominators: computed, ≈6e-5 relative — provably negligible; the unknown-denominator angle is CLOSED, not hand-waved.
