# Your ensemble gain is probably variance reduction — and there's a cheap test that tells you

*Draft, 2026-07-29 (S6E7 Day 22). Numbers from `results/probe_mslope.txt`,
`probe_mslope_phase2.txt`, `probe_mslope_te13_honest.txt`.*

## The setup

Twenty-one days into a Kaggle Playground campaign, the deployed submission was an 8-leg
logistic-regression stack over 8-seed model breadths, sitting at 0.95081 OOF balanced
accuracy. Eighteen consecutive "add/replace a leg" experiments had been vetoed. On day 21
a pattern was noticed and written down as a screening rule:

> **Rule #3**: a lever measured at one ensemble member is worth ≈1/M of its measured size
> at the deployed breadth. Screen at the deployed M, never at M=1.

It came from two levers — top-k epoch snapshot averaging and a tuned config basin — whose
advantage decayed roughly as `M^-1` as members were averaged. It looked like a law.

It is a law about *some* levers. Generalising it to all of them was the mistake, and the
mistake is the interesting part, because the rule as stated prices out exactly the class
of lever you actually want to find.

## The problem with the evidence

Here is the strongest evidence one of those levers presented:

```
per-seed delta:  s42 +0.000206   s7 +0.000288   s123 +0.000281   s2024 +0.000213
                 mean +0.000247   se 0.000022   4/4 positive
```

Eleven standard errors. Replicated at every seed. In most workflows that closes the case.

And at M=4 the advantage was +0.000080 — essentially gone.

The resolution: **a variance reducer is significantly positive at every individual member.
That is what a variance reducer does.** It lowers each member's posterior noise, so it wins
at every seed, with a tiny standard error, and it is worth nothing once you average eight
members — because averaging removes the same noise, from the *incumbent* arm too.

Per-seed significance cannot distinguish "this adds information" from "this removes noise
I was about to remove anyway." So what can?

## The instrument

Measure the delta at M = 1, 2, 3, 4 and fit the exponent of `|delta| ~ M^b`. The claim is
that `b` identifies the mechanism:

| mechanism | why | predicted `b` |
|---|---|---|
| **uniform mean shift** (real information) | the shift survives averaging by linearity | **0** |
| **selection noise** (you picked the sign at M=1) | difference of two means, CLT | **−0.5** |
| **variance reduction** | competes directly with averaging | **−1** |

That is a prediction, not a measurement, so it needs controls — and the controls have to be
**magnitude-matched**. An exponent difference measured between a +0.0009 lever and a
+0.0003 lever tells you about magnitude, not mechanism.

Three controls, all built from per-seed OOF arrays already on disk, all calibrated by
bisection to read ≈+0.0003 at M=1 (the size of the levers under suspicion):

- **C1, known-real**: mix every member toward `onehot(y)`. A uniform mean shift toward the
  truth. *This consumes the labels and is quarantined instrument calibration — it can never
  become a lever.*
- **C2, known-fake**: split the 8 seeds into two disjoint halves and call one of them "the
  challenger", **choosing which one by the M=1 read**. That reproduces the winner's-curse
  selection every real screen performs. Repeated over many partitions, this gives the
  **null distribution of the exponent** — and therefore its MDE, without which "−0.71"
  cannot be distinguished from zero.
- **C3, known-fake**: shrink each member toward the average of a *disjoint* set of members.
  Identical mean, lower variance. Pure variance reduction. (Disjoint matters: shrinking
  toward an average that contains the member forces the delta to zero at full M by
  construction, which would fake the result.)

Metric held fixed throughout: balanced accuracy under decision weights fitted **once** on
the full breadth and then frozen, so no read is secretly a re-tuning of the decision layer.

## The result

```
C1  uniform mean shift        exponent -0.027 +/- 0.012
C2  selection noise           exponent -0.417 +/- 0.178      null band [-0.773, -0.060]
C3  variance reduction        exponent -1.002 +/- 0.086
```

All three land on their point predictions. C1 and C3 separate by **11.2 standard
deviations**. The instrument works.

Applied to the two levers that generated rule #3 in the first place:

```
snap2 / snap3 / snap4    -1.53 / -0.91 / -0.77     variance reduction
ji                       -0.71                     inside the null band
```

Snapshot averaging reads as variance reduction — which it mechanically *is*, so the
instrument agrees with the mechanism it should. The config-basin lever, whose 11-sigma
per-seed evidence looked like a genuine mean shift, is **indistinguishable from selection
noise**.

## The prediction that failed

I predicted that shrinking members toward a *better* posterior — the deployed stack —
would read as a mean shift, and offered it as the label-free version of C1. It read
**−1.086**. The reason is the finding, not an exception to it: the stack is itself an
average over 64 model fits, so "shrink toward something better" is *mechanically a variance
reducer*. You cannot get a mean-shift control for free by pointing at a stronger model.

## What it changes

**Rule #3′, replacing rule #3**: screen by the exponent, then by the magnitude.

- `b ≈ 0` → information. Worth its **full M=1 size** at any deployed breadth. Escalate on
  size alone.
- `b ≈ −0.5` → indistinguishable from selection noise. Dead.
- `b ≈ −1` → variance reduction. Substitutes for breadth averaging. Dead at M ≥ 8.

Two things about this are better than the rule it replaces. It is **sharper** — the old
rule would have discarded a genuine information lever, since it assumed the 1/M decay was
universal. And it is **cheaper**: four members and seven minutes of numpy against arrays
you already have, instead of rebuilding a candidate at the full deployed ensemble size.

## The honest bottom line for the competition

None of this moved the submission. The deliverable bar was +0.001 balanced accuracy at the
deployed ensemble size; nothing measured today came within a factor of eight of it, and the
finals were left exactly where they were. What changed is the *screening rule* — and the
recognition that the campaign's own standard of evidence, "it replicated at every seed with
a tiny standard error," is precisely what a worthless lever looks like.
