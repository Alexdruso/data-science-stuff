# 12th place — how the private leaderboard undid the public one

I finished **407th on the public leaderboard and 12th on the private one** (2,817 teams). I would love to tell you this was a masterstroke. Most of it was the shakeup doing my work for me. But the one decision that was mine, I got right, and it's the only part worth writing down.

Quick problem recap: three classes (GALAXY 65%, QSO 20%, STAR 14%), balanced accuracy, ten mostly-photometric features plus redshift. Redshift alone almost solves it (STAR ≈ 0.07, GALAXY ≈ 0.51, QSO ≈ 1.88 in the mean), so everyone lands near the same score and the whole competition is a fight over the last 0.001.

## The plateau, and what actually broke it

I started where everyone starts: LightGBM on raw features (0.9559 CV), `class_weight="balanced"` for the rare STAR class (+0.008, the single biggest honest jump of the whole run), a tuning pass (0.9657). Then I stacked a handful of GBDTs and hit **0.9662, and sat there.** Adding another tuned LGBM did nothing. Adding a tuned XGBoost did nothing. The GBDT cluster had said everything it had to say.

![plateau](results/analysis/fig_plateau_break.png)

What broke it was not strength, it was **difference** — and not "a different tree library." I ported three ideas (credit to cdeotte's public notebooks) that each departed from the GBDT-on-raw-features mold on a different axis:

- **a different feature space**: a 240-feature XGB with fold-safe quantile-bin target encoding → 0.9671
- **a different loss and model class**: a from-scratch RealMLP with a logit-adjusted loss, a neural net that optimizes the imbalanced objective directly instead of reweighting → 0.9690, and about +0.0019 in the stack, the biggest lift after `class_weight`
- **a different decomposition**: a chain-cascade that factors P(class) as P(QSO) × P(STAR | not-QSO), two binary CatBoosts instead of one 3-way head

Those three dragged the stack to ~0.9705. Every one added information the GBDTs did not have. Every same-family model I tried after (more XGB variants, HPO on the Deotte bases, seed bags) added nothing the linear stacker couldn't already reconstruct. The lesson I'd tattoo on next season: **a stacker rewards a model that is wrong differently, not a model that is slightly more right.**

## The one decision that mattered

By the end I had three finalists, all within 0.0003 of each other on public: an LR stack, a GBDT stack, and a `metablend` that averages the metas. My *best public score* was a metablend at **0.97119**. It sat at the top of my own board and it was tempting to just click it.

I didn't. My cross-validation kept whispering that the metablend's public edge was partly in-sample: the calibration and threshold weights it fit had the smallest CV→LB gap of anything I ran, the classic fingerprint of a submission that has memorized the public split. So I selected two later, more heavily-vetted metablend regenerations at 0.97108 public instead, and left the shinier number on the table.

The private leaderboard was blunt about it:

![shakeup](results/analysis/fig_shakeup.png)

The 0.97119 pick I *didn't* select fell to **0.97033, the worst private transfer of the four.** The finals I kept scored **0.97041** (12th). And the GBDT stack I'd half-dismissed for its lower public score, 0.97094, quietly posted 0.97042 on private, better than either final I actually selected. Public rank near the top wasn't just noisy, it was *anti-predictive*: the higher a submission sat publicly, the harder it fell.

None of this is surprising once you look at where the field actually was:

![density](results/analysis/fig_lb_density.png)

Thirty-one teams sat within ±0.0001 of my public score, about fifteen places for every ten-thousandth of a point. A dense wall of teams piled up at the very top of the public board around 0.9725. When the private split landed, a lot of that wall was standing on overfit, and it came down. I climbed 395 places without touching my model; I just wasn't in the wall.

## Honest bookkeeping

Let me be clear about the split between luck and skill. **The 395-place jump was luck**, a favorable shakeup that hit the public-LB-chasers harder than me. **Trusting CV over the public score was skill**, in the narrow sense that it's a repeatable habit and it was worth real rank here.

Things I tried that went nowhere, so you don't have to:

- **More GBDT tuning / Optuna on strong bases.** Flat at the stack, every time.
- **A denoising autoencoder** for learned features scored worse than raw. It optimizes reconstruction, which blurs the exact low-redshift STAR/GALAXY boundary the labels live on.
- **kNN retrieval against the real SDSS catalog.** The real neighbors make the *same* mistakes on the hard rows; the overlap is intrinsic to photometry, not a binning artifact.
- **`alpha`/`delta` exact-value "lineage" features** (the synthetic generator repeats coordinates): real signal in-sample, zero transfer through any consumer.
- **Low-z specialist and cost-matrix decision rules.** Pure operating-point sliding, no new ranking information.

The wall at the top is real and I'm not above it. The gap from my 0.9704 to the genuine frontier needs information this dataset doesn't contain (morphology, not just photometry) or a stronger single extractor than I had. But if the choice is between the number that flatters your public rank and the one your CV trusts: **trust the CV.** The 0.00008 between the finals I kept and the one I was tempted by doesn't sound like anything, but on a board this dense it's the gap between a placing I'm happy with and one I'd have quietly regretted. It cost me a prettier public score, and it's the only trade in the whole run I'd make again without thinking.
