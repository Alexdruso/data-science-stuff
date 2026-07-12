# Day-10 PM advisor reports (2026-07-12) — two Fable 5 agents, condensed

User-directed: two advisor agents (adversarial research strategist + endgame
decision theorist) reviewed the full campaign state. Actioned same evening; this
file is the condensed record + the Day-11 backlog.

## Decision theorist (endgame audit)

- **Ran the repo's own mc_cells over the real artifacts (read-only)**: the planned
  pair lrstack_u3 + lrstack6b_r has hedge value **+0.00000 at face** (E[priv]
  0.95084 vs 0.94946, sd(Δ)=0.00022, P(u3 wins)=1.000). Hedge pays only if the
  repaired candidate's OOF understates its true mean by **δ_r ≥ ~0.0012** — the
  knife-edge sits inside the untested "reads 0.001-0.002 low by construction" band.
  CONFIRMED same evening by `probe_finals_bracket.py`: crossover between
  δ_r=+0.0008 and +0.0016 vs the all-u pair (u3+u2). 6b_r's #2 seat is justified as
  fat-tail insurance (Day-2 v2 precedent), not by the MC.
- **Seeds are spent**: per-seed sd ~1e-4, 5-seed residual ~4e-5 vs 6e-4
  private-split sd. No more accuracy-motivated seeds.
- **Tiebreak-theorem scope correction**: mc_cells conditions on OOF cells as truth,
  excluding epistemic mean uncertainty by construction — public reads CAN carry
  information about instrument bias across surfaces. One read qualifies now:
  sign + rough magnitude of pub(lrstack_u3_r) − pub(lrstack6b_r) (~3σ on δ_r).
  Rule: request reads only when hypothesis gap ≥ 2× read noise (≥0.0009 paired /
  ≥0.003 absolute).
- **Ops (DONE same evening)**: off-box backup tarball + sha256 manifest (results/
  and submissions/ are gitignored on a randomly-rebooting box); PROTOCOL additions:
  explicit Kaggle finals selection ≥48h before deadline (auto-select = the gamed
  public band), freeze finals-lineage tags ~Jul 27.
- **Allocation insight**: the campaign's largest gain (+0.0014) came from a forum
  post; 6 days of internal probing netted +0.0009. Standing item: short daily
  forum/notebook sweep. Guard against action bias — internal experiments now have
  ~0 mean EV and nonzero artifact-contamination tail risk.

## Adversarial strategist (saturation-thesis attack)

- Verdict: saturation proven for architectures/combiners/missing-driver region;
  NOT proven for the **joint boundary-band channel**. Field notebook published
  2026-07-12 (r1ck.sync "Reverse Engineering the Generator"): the generator
  learned a SIGMOID at the sleep 6/7 cut points (sleep 6.01-6.10 × stress=high =
  20-40% unhealthy); honest nested solo 0.95054 via kNN label-distribution
  features in the 6 leakage numerics. Our Day-10-AM amplifier pricing conditioned
  at rule-triple resolution — too coarse.
- Scoping errors found in our kills: te3p ranked pairs by |ΔCramérV| (generator
  couplings), never testing the RULE pairs at exact-value resolution; combo2 used
  sleep BUCKETS. Balanced-class-weight training = unexamined Day-1 global (field
  warns it double-corrects with the decision layer). The two-surface hedge is
  recipe-confounded (lrstack6b_r has zero te13 legs).
- Ranked directions: #1 kNN label-distribution features (field-verified +0.0003);
  #2 rule-pair TE exact-value; #3 unweighted-training leg (seat-shaped); #4
  te13×repaired lane (de-confounds the hedge); #5 last-week variance program.

## Same-evening outcomes

- **#2 rule-pair TE: FLAT, measured** (`hgbc_te3rp_u_s42` 0.9503 vs 0.9504;
  ~250 rows/cell, healthy counts). The sigmoid band does not pay through pair TE.
- **#1 kNN features**: `knn_features.py` built (fold-safe drop-self GPU kNN,
  ~110 s/fold, cached knn_u_s42_f*); consumer `hgbc_te3knn` in the CPU chain.
- **#3 unweighted leg** (`hgbc_te3ncw`, S6E7_NO_CW=1) + **#4 te13×repaired lane**
  (hgbc_te3_r_s42 CPU; realmlp_te13_r seeds 42/7/123 GPU) launched; stack rebuild
  and seat tests = Day-11 morning.
- δ-bracket MC: `results/probe_finals_bracket.txt` (knife-edge δ_r* ≈ 0.0012).

## Day-11 backlog

1. Read overnight: hgbc_te3knn (kill: < +0.0002 solo AND seat veto), hgbc_te3ncw
   (seat-shaped), hgbc_te3_r + realmlp_te13_r ×3.
2. Rebuild the repaired te13 stack (lr over {hgbc_te3_r, realmlp_te13_r breadth} +
   old r-legs) → intersection read vs lrstack_u3 → the deployed-level δ_r
   estimate → finals decision via the bracket table.
3. Seat tests for any solo ≥ +0.0002 (12-cell, probe_combiner_ftt pattern).
4. Ask user for the paired public read (u3 vs 6b_r, sign + rough magnitude).
5. Standing: daily forum sweep (10 min, kaggle CLI kernel pulls).

## Discarded advisor items (with reasons)

- TabPFN-3 / specialist binaries / snapshot averaging / missingness TTA — priced
  ~0 by both advisors (architecture/inference variants; five-veto lesson).
- Any label curation — Day-10 AM inversion is decisive.
- More seeds for accuracy — spent (see above).
