# Day-18 brainstorm (2026-07-22) — 3 fresh lenses

Prior lens sets: representation / model-training / log-red-team (Day-13-eve),
information-pathway / loss-geometry / evaluation-asymmetry (Day-17). Screened
against the kill list: combiner matrix COMPLETE (Day-13), decision layer proof
(Day-14), dispersion seat veto #11, fold-count closed (Day-15/16), stack depth
washes u2→u5, reconstruction channel vetoes #14/#17.

Revival rule applied (premature-ceiling feedback): Day-10 advisors PRICED OUT
snapshot averaging + missingness TTA ("five-veto lesson") — priced ≠ measured,
and the Day-12+ arm harness has since cut the canary cost to ~100 s.

## Lens A — prediction-aggregation geometry (pointwise pooling, NOT the meta)

The combiner matrix swept the META (family, C, objective, calibration,
structure) over FIXED leg arrays. Every leg array itself is an ARITHMETIC mean
of member softmaxes (breadth seeds; fold means inside zoo_cv). The pooling
FUNCTION is an unswept axis: arithmetic mean is optimal for variance under
squared loss, but argmax/weighted-argmax consumers care about the boundary —
log-space (geometric) pooling weights confident members differently, median /
trimmed kill member outliers.

- **A1 — MEASURED NULL (same day)**: repooled breadth members {logit-mean,
  median, trimmed} for realmlp_te13_u (12 seeds) + hgbc_te3_u (8 seeds),
  paired argmax + robust-wt vs the arithmetic incumbent
  (`probe_pool_geom.txt`): every arm within ±0.00005, both families (best
  trimmed +0.00002 wt) vs bar +0.0007. Arithmetic-mean variance-optimality
  holds at the boundary too; **pooling-function axis measured-closed**, no
  escalation.
- A2. Rank-averaging across members — SCREENED OUT: rank pooling destroys the
  proba scale the decision weights act on; equivalent to a monotone
  recalibration the per-class-power cell already measured (+0.00004).

## Lens B — training-trajectory marginalization (weight/epoch/input space)

The HP axes (single + joint TPE) optimize a POINT on the training trajectory;
best-epoch selection at epochs 3-7 on a steep trajectory is a high-variance
point estimate (Day-10 persona #6, P .12, priced out unmeasured).

- B1. epoch-window softmax averaging — SCREENED OUT ON RECORD CHECK: this IS
  the package's `snapshot_k` knob (mean softmax over top-k epochs,
  realmlp.py:789), already "measured, NOT re-swept: snapshot_k (wash)"
  (tune_realmlp_td.py header, Day-12). Not a priced-out cell after all — a
  measured one. Struck.
- B2. inference-noise TTA — SCREENED OUT ON CONFIG CHECK: the deployed CFG
  trains with `numeric_noise_std=0.0` (train_realmlp_td CFG + package
  default), so there is no train-time noise distribution to marginalize;
  TTA would inject noise the model never saw. The noise-0.02 config lives in
  the joint-TPE basin already closed at 5-fold (veto #15). Struck.
- B3. SWA/EMA weight averaging — remains the one unmeasured cell of this
  lens, but B1's wash (prediction-space averaging over the same trajectory)
  guts the prior and the wiring is real (optimizer hook in the shared
  package). DEFERRED, not dead — revive only on a fresh mechanism argument.
- **Lens B net: no runnable survivor — the screen killed it honestly (two
  record checks, zero compute). GPU freed for tabms breadth depth instead.**

## Lens C — transductive test-use (the 295k unlabeled rows as training signal)

Never measured on s6e7: distillation-to-student (Day-9, replaced the model —
dead) and external-50k augmentation (off-manifold — veto) are DIFFERENT cells.
Pseudo-labeling keeps the model and augments its TRAIN folds with on-manifold
rows the adversarial-shift diagnosis says we under-cover.

- **C1 — MEASURED NEGATIVE (same day)**: confident-test pseudo-labels
  (τ=0.99 → 32,204 rows, 10.9% of test, class dist [1.8k AR, 12.5k fit,
  18.0k unh], 9% missing-driver) into the hgbc_te3 canary, TE surface held
  at the twin's: **argmax 0.95003 (−0.00031) / wt 0.95012 (−0.00014)**
  (`probe_pseudo_hgbc.txt`). Confound noted post-read: the minority-heavy
  pool shifts what class_weight="balanced" computes → weight-matched
  deconfound arm (S6E7_PSEUDO_MATCH=1, explicit real-fold balanced weights,
  pseudo mass added in real class shares) run same day: **+0.00019 wt (sign
  flip)** — then the s7 replication read **−0.00015 wt** (2-seed mean ≈ 0).
  The confound fix was correct (raw negative = artifact) but the deconfounded
  truth is ZERO: **transductive cell measured-FLAT, closed.** Reports:
  `probe_pseudo_hgbc{,_match,_match_s7}.txt`.
- C2. Adversarial-weighted pseudo-labels (test-like rows only) — FOLD INTO C1
  as a diagnostic split (test-like vs not), not a separate run.

## Measurement order today

1. A1 now (CPU array math, minutes).
2. C1 after the tebag probe frees the CPU (~15 min).
3. Lens B: no survivor (see above) — GPU goes to tabms_te13 seeds s2025/s31
   to deepen the #2-menu hedge breadth (queued item 2) instead.
Survivors ≥ bar get the standard escalation; everything else recorded here.
