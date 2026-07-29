# Averaging reveals information and destroys noise — a cheap test for which one you found

*Draft, 2026-07-29 (S6E7 Day 22). Numbers from `results/probe_mslope.txt`,
`probe_mslope_te13_honest.txt`, `probe_mslope_phase2.txt`.*

## The setup

Twenty-one days into a Kaggle Playground campaign, the deployed submission was an 8-leg
logistic-regression stack over 8-seed model breadths, at 0.95081 OOF balanced accuracy.
Eighteen consecutive "add or replace a leg" experiments had been vetoed. On day 21 a
pattern was noticed and written down as a screening rule:

> **Rule #3**: a lever measured at one ensemble member is worth ≈1/M of its measured size
> at the deployed breadth. Screen at the deployed M, never at M=1.

It came from two levers whose advantage decayed roughly as `M^-1`. It looked like a law.

It is a law about *some* levers, and generalising it was the mistake — because as stated it
prices out exactly the class of lever you want to find.

## The problem with the evidence

Here is the strongest evidence one of those levers presented:

```
per-seed delta:  s42 +0.000206   s7 +0.000288   s123 +0.000281   s2024 +0.000213
                 mean +0.000247   se 0.000022   4/4 positive
```

Eleven standard errors, replicated at every seed. In most workflows that closes the case.
At M=4 the advantage was +0.000080 — essentially gone.

The resolution: **a variance reducer is significantly positive at every individual member.
That is what it does.** It lowers each member's posterior noise, so it wins at every seed
with a tiny standard error — and it is worth nothing once you average eight members,
because averaging removes the same noise from the *incumbent* arm too.

Per-seed significance cannot distinguish "adds information" from "removes noise I was
about to remove anyway." So what can?

## The instrument

Measure the delta at M = 1, 2, 3, 4 and fit the exponent of `|delta| ~ M^b`.

Predictions, before measuring:

| mechanism | why | predicted `b` |
|---|---|---|
| uniform mean shift (information) | survives averaging by linearity | 0 |
| selection noise (sign picked at M=1) | difference of two means, CLT | −0.5 |
| variance reduction | competes directly with averaging | −1 |

That needs controls, and they must be **magnitude-matched** — an exponent difference
between a +0.0009 lever and a +0.0003 lever is confounded with size. Three controls, built
from per-seed OOF arrays already on disk, each calibrated by bisection to ≈+0.0003 at M=1:

- **C1, known-real**: mix every member toward `onehot(y)`. *Consumes labels — quarantined
  instrument calibration, can never be a lever.*
- **C2, known-fake**: split the seeds into disjoint halves and call one "the challenger",
  **choosing which by the M=1 read** — reproducing the winner's-curse selection every real
  screen performs. Over many partitions this gives the **null distribution** of the
  exponent, and hence its MDE.
- **C3, known-fake**: shrink each member toward the average of a **disjoint** set of
  members. Identical mean, lower variance. (Disjoint matters — shrinking toward an average
  containing the member forces delta→0 at full M by construction and fakes the result.)

```
C1  uniform mean shift        -0.027 +/- 0.012
C2  selection noise           -0.417 +/- 0.178     null band [-0.773, -0.060]
C3  variance reduction        -1.002 +/- 0.086
```

All three land on their point predictions; C1 and C3 separate by **11.2 sd**.

## The real-lever test, and the surprise

Synthetic controls prove the exponent *can* separate constructed mechanisms. The question
is whether it separates real ones. Two levers with enough seeds on disk, read with an
honest per-arm decision layer (weights fitted on one half, scored on the other):

| lever | M=1 | M=2 | M=3 | M=4 | exponent |
|---|---|---|---|---|---|
| **the target-encoding recipe** (known-real, +0.0004, the campaign's single largest find) | +0.000216 | +0.000354 | +0.000391 | — | **+0.558** ± 0.138 |
| **the config-basin lever** (the 11-sigma one above) | +0.000295 | +0.000201 | +0.000133 | +0.000055 | **−1.107** ± 0.458 |

The real information lever sits **4.5 sd above the null band**. The 11-sigma lever sits
below it, 3.6 sd away from the real one. The instrument separates them cleanly.

**The surprise is the sign.** The real lever's advantage does not merely survive averaging
— it **grows**, monotonically, +0.000216 → +0.000391. That sharpens the whole picture:

> **Averaging reveals information and destroys noise.**
> At M=1 both arms are noisy, and member noise randomly throws boundary rows to the wrong
> side, *masking* a systematic edge. Averaging removes the mask, so a genuine advantage is
> more fully expressed at M=8 than at M=1.

C1's ≈0 is the *oracle* case — a uniform shift toward the truth is already fully expressed
at one member, so it neither grows nor shrinks. A realistic information lever, partly
masked by member noise at M=1, reads **b > 0**. Revised scale: **b > 0 information ·
b ≈ −0.5 selection noise · b ≈ −1 variance reduction.**

## The prediction that failed

I predicted that shrinking members toward a *better* posterior — the deployed stack — would
read as a mean shift, and offered it as a label-free version of C1. It read **−1.086**. The
reason is the finding, not an exception to it: the stack is itself an average over 64 model
fits, so "shrink toward something better" is *mechanically a variance reducer*. You cannot
get a mean-shift control for free by pointing at a stronger model.

## What it changes

**Screen by the exponent, then by the magnitude.**

- `b > 0` → information. Worth its M=1 size *or more* at the deployed breadth.
- `b ≈ −0.5` → indistinguishable from selection noise. Dead.
- `b ≈ −1` → variance reduction. Substitutes for breadth averaging. Dead at M ≥ 8.

Better than the rule it replaces on both axes. **Sharper**: the old rule assumed the 1/M
decay was universal and would have discarded a genuine information lever. **Cheaper**: four
members and minutes of numpy against arrays you already have, instead of rebuilding a
candidate at full deployed size.

One implementation note that mattered more than expected. The obvious way to fit per-class
decision weights — Nelder-Mead on `balanced_accuracy(argmax(p·w))` — is fitting a
**piecewise-constant** objective. It can never reach a tight tolerance, so it burns its
whole iteration budget (~45k evaluations per call) and turned a five-minute read into a
multi-hour one. A coarse-to-fine grid over the two free log-weights was ~30× faster *and*
found a strictly better optimum (+0.000077 balanced accuracy on the deployed ensemble).
If your inner loop optimises a step function, do not use a simplex method.

## The honest bottom line

None of this moved the submission. The deliverable bar was +0.001 balanced accuracy at the
deployed ensemble size; nothing measured came within a factor of eight, and the finals were
left where they were. A structurally novel leg built the same day — an explicit Bayes
factorisation `P(y|x) = Σ_d P(d|x)P(y|d)` over the masked causal drivers — turned out
*worse* on exactly the rows it was designed for, because under MCAR the trees already
estimate the same quantity directly.

What changed is the screening rule, and the recognition that the standard of evidence most
ensemble work relies on — "it replicated at every seed with a tiny standard error" — is
precisely what a worthless lever looks like.
