# Day-17 brainstorm (2026-07-21) — three fresh lenses

Lenses already spent on prior days (not reused): training-dynamics,
decision-theory-under-uncertainty, red-team-the-instruments (Day-14);
representation, model-training, log-red-team (Day-16); external-signal /
objective channel taxonomy (Day-11). Every idea below is screened against
(a) the 3-dim label-channel model (conditional on stress x activity x exact
sleep, flips are independent of everything — te13 saturates it at ~1,900
rows/sleep value), (b) 13 seat vetoes / saturated combiner axis, (c) the
decision-layer closure-by-proof, (d) the joint-shift blind-spot ledger.

## Lens 1 — information-pathway (where could residual bits still flow?)

The archaeologist kill-shot CONDITIONS ON EXACT SLEEP, so its "label channel
is exactly 3-dimensional" proof covers complete-driver rows only. The
missing-driver regime (26% of rows, 86% of residual error) is governed by
P(label | observed pattern) — and the generator ADDED cross-column dependence
(orig |corr| 0.003 -> synth ~0.10), so observed non-driver columns carry real
signal about the MISSING driver value.

- **P1. Explicit driver reconstruction (SURVIVOR)** — train a fold-safe
  regressor sleep_duration ~ other 12 cols on complete fold-train rows;
  feed [imputed value, rule-bucket one-hots of the imputed value,
  imputed x is-missing interaction] to hgbc_te3. What te13 does NOT span:
  te13 is a set of PER-COLUMN posteriors; the reconstruction is a JOINT map
  into the rule's own coordinate. kNN joint-density was flat, but it estimated
  label density, not the driver value. Prior LOW (trees see all columns
  jointly already), cost ~15 min CPU, cleanly falsifiable paired 5-fold.
- P2. Naive-Bayes product-of-TE-posteriors feature — measured flat Day-10
  (NB-prior). SCREENED OUT.
- P3. Impute stress/activity too — stress near-deterministic, activity
  low-card; partial-pattern TE already carries their pattern posterior.
  Fold into P1 only if the sleep leg reads positive.

## Lens 2 — loss-geometry (unswept loss-SHAPE knobs, distinct from the HP axis)

The Day-12 ladder + Day-17 joint TPE sweep optimizer/architecture axes; the
S2 arm swept the objective FAMILY. Two knobs that change the loss SHAPE
within CE were never measured on s6e7 (both live in the package, default-off):

- **L1. focal_gamma {0.5, 1.0} (SURVIVOR)** — loss curvature over hard
  examples; the flip rows ARE the hard examples and they're learnable
  cell-clustered structure, so focusing could re-weight exactly them.
  Fold-1 ladder arms, ~100s each, paired vs cached base.
- **L2. sample_weight_power {0.3, 0.6} (SURVIVOR)** — resampling vs
  loss-adjustment (balanced softmax); s6e6 prior says oversampling lost to
  logit adjustment, but it was never measured HERE and it composes with lpp.
  Same instrument, ~100s each.
- L3. class_weight_power — same mechanism class as lpp (already at its
  measured optimum 1.075; lpp1.0/1.2 both read); redundant. SCREENED OUT.

## Lens 3 — evaluation-asymmetry (what can we affect that OOF can't see?)

- E1. Test-transform TE bagging — OOF-invisible, recorded UNREADABLE Day-16.
  Still excluded by design (LB-blind: no offline read exists). SCREENED OUT.
- E2. Exact-duplicate / profile-group pooling across train∪test — dead by
  measurement: test->train NN frac<0.10 = 0.0000 (Day-11, no near-dupes).
  SCREENED OUT.
- E3. Per-fold vs global decision weights — LOFO optimism +0.00003 (Day-10);
  decision layer closed by proof (Day-14). SCREENED OUT.
- E4. Rule-overwrite on determined test rows — anti-motivated by the noise
  curation inversion (baseline already recovers 17.2% of rule-contradicted
  rows; flips are learnable structure an overwrite would destroy). SCREENED
  OUT without compute (measured-adjacent, expected negative by mechanism).

## Survivors measured today (Day-17 pm)

1. P1 driver-reconstruction -> `probe_impute_driver.py`, hgbc_te3 canary,
   full 5-fold paired, screen bar +0.0007 (house 1-seed canary convention).
2. L1+L2 -> four `tune_realmlp_td.py` STAGE_L arms after the joint TPE frees
   the GPU (focal0.5, focal1.0, swp0.3, swp0.6), fold-1 dw paired, bar +0.0007.
