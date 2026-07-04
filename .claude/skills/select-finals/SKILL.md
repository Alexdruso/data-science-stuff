---
name: select-finals
description: Select the final two Kaggle submissions at the end of a competition — rank candidates by CV (not public leaderboard), detect public-split overfit, and pick diverse finals. Use when asked to choose/select final submissions, e.g. "which two submissions should we select for s6e6?", "pick the finals", "the deadline is tomorrow, what do we submit?".
---

# Select the final submissions

Use this at competition end, when Kaggle asks for (usually) two finals. This decision is worth
real rank and is easy to get wrong by clicking the best public score. The playbook below is
distilled from s6e6, where it was the single decision that mattered: the tempting best-public
pick (0.97119) fell to the **worst** private transfer (0.97033), while the CV-vetted finals held
(0.97041 → 12th of 2,817, +395 places on the shakeup).

## Principles

1. **Rank candidates by OOF CV, not public LB.** On a dense leaderboard (dozens of teams per
   0.0001) the public split is a small noisy sample; near the top, public rank can be
   *anti-predictive* — the submissions that sit highest publicly are disproportionately the ones
   that overfit the public split, and they fall hardest on private.
2. **Know the overfit fingerprint.** A submission whose fitted post-processing
   (calibration, per-class threshold weights, blend weights re-tuned after seeing the LB) shows
   a *suspiciously small CV→LB gap* relative to its siblings has likely memorized the public
   split. Its public edge is in-sample; discount it.
3. **Pick two diverse finals**, not the two shiniest numbers — e.g. the stack/ensemble plus a
   structurally different runner-up (a GBDT stack vs an LR stack, or best single model). Two
   near-identical regenerations waste the second slot's hedge value.
4. **If forced to a single pick, prefer the ensemble/stack** over the best single model at a CV
   tie — it averages out per-model error and is the safer private-LB bet.
5. **Accept leaving a prettier public number on the table.** A ~0.0001 public sacrifice for a
   submission your CV trusts is the correct trade on a dense board.

## Steps

1. **Assemble the candidate table** from `results/cv_scores.csv` + the competition `CLAUDE.md`
   experiments log + `kaggle competitions submissions -c <id>`: for each candidate, OOF CV,
   public LB, and the CV→LB gap.
2. **Flag overfit suspects**: candidates whose gap is an outlier (small) versus the batch,
   especially those with post-hoc fitted weights or many LB probes behind them.
3. **Rank by OOF CV** among the non-flagged candidates; break near-ties (within CV noise,
   ~1 fold-std) by preferring (a) ensembles/stacks over singles, (b) structural diversity
   between the two picks.
4. **Select on Kaggle only when the user confirms** the two picks.
5. **Record the decision** in the competition `CLAUDE.md` (which two, why, what was rejected and
   why) — after the private LB lands, write the CV→public→private table into the log so the
   next competition's selection has evidence.

## Honest bookkeeping

Distinguish luck from skill when reporting the outcome: a favorable shakeup is luck; trusting
CV over the public score is the repeatable habit. Don't let a lucky jump (or an unlucky drop)
revise the process.
