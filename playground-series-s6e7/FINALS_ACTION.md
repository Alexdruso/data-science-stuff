# S6E7 — WHAT YOU STILL HAVE TO DO

*One page. Everything else about this competition is finished and frozen
(`git tag s6e7-freeze-2026-07-27`). Full reasoning lives in `CLAUDE.md`; you do not need
to read it to act on this.*

---

## 1. Click these two submissions on Kaggle — deadline ~**July 31**

| slot | file | sha256 (first 16) |
|---|---|---|
| **#1** | `submissions/lrstack_u4_r.csv` | `ed5a9bfff4b6ed51` |
| **#2** | `submissions/realmlp_te13_u_breadth.csv` | `8fd6904ef65c697b` |

Go to the competition's **My Submissions** page and tick these two for final scoring.
Both are already uploaded and scored — you are only selecting them, not resubmitting.

**Why this is not optional.** If you select nothing, Kaggle automatically picks your two
best **public** scores. About 48 of the submissions on that page are `lbprobe_*` files that
were deliberately hill-climbed against the public split — they are *designed* to score well
publicly and mean nothing privately (measured: +0.00076 of public score was fakeable in
three casual rounds). Auto-select would very likely hand in a probe artifact. This is
exactly how s6e6 went wrong: the best-public pick took the worst private transfer.

Do it at least 48 h before the ~August 2 deadline, in case the UI or your login misbehaves.

---

## 2. Copy the backup off the box

```
~/s6e7_finals_backup_2026-07-27.tar.gz      (139 MB)
```

`results/` and `submissions/` are gitignored and exist only on a machine that hard-reboots
at random ([[project_pc_random_reboots]]). The tarball holds the finals CSVs, their OOF/test
arrays, decision weights, a sha256 manifest and `CLAUDE.md`. It has been test-restored from
a clean extract — 25/25 hashes verify — so it is known-good, not assumed-good.

Kaggle keeps the submitted CSVs regardless, so your *selection* survives total box loss
either way. The tarball protects the ability to rebuild and explain the result.

---

## What you are handing in

- **#1 `lrstack_u4_r`** — an 8-leg LR stack (4 GBDT families + RealMLP + LGBM/HGBC on the
  te13 target-encoding recipe + an MLP), 8-seed legs, 10-seed outer meta, then a tuned
  decision layer. Honest OOF balanced accuracy **0.95081**; mean-dominant over every other
  candidate on every honest instrument by 1e-4 to 1e-3.
- **#2 `realmlp_te13_u_breadth`** — a plain 8-seed RealMLP breadth, no fitted combiner.
  It is the only eligible hedge with real tail mass (644 disagreeing rows, P(beats #1) ≈
  0.25). It insures against the one part of #1 that is fitted on OOF — the meta and
  decision layer — failing to transfer.

Chosen by cross-validation, never by leaderboard position, per the LB-blind protocol.

---

## If you want to change your mind

The only defensible substitution is **`lrstack6b_r`** in the #2 slot, if you would rather
hedge the *feature recipe* than the *meta layer*. It costs ~4e-4 of expected mean and buys
a larger disagreement set (1,426 rows). It is a taste call, not a correctness one:
`probe_rank_regime` measured that the #2 choice moves your probability of beating any given
rival by **≤3 percentage points in every regime**. The #1 slot is the whole game.

**Do not** select `champion_v1` — it insures against a failure mode that was tested and
refuted, at ~1.5e-3 of expected mean.
