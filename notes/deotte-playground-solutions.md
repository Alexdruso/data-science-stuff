# Deotte's public Playground solutions vs this repo

Source: [`cdeotte/kaggle-playground-solutions`](https://github.com/cdeotte/kaggle-playground-solutions)
(commit `e5a03ff`, 2026-11-03), announced in the
[S6E11 discussion](https://www.kaggle.com/competitions/playground-series-s6e11/discussion/756276). Also read:
[`cdeotte/Tabular_Foundation_Models`](https://github.com/cdeotte/Tabular_Foundation_Models) and the NVIDIA blog post
[Winning a Kaggle Competition with Generative AI-Assisted Coding](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)
(the S6E3 agent workflow). The Kaggle writeups themselves are rendered by JavaScript and could not be read; everything
below comes from the repo READMEs and code, which Deotte says were cleaned up and **not re-run**.

## 1. The gap, on the competitions we both played

| Comp | Our best (honest OOF AUC) | Deotte stack CV | Deotte best single model | Gap (stack) |
|---|---:|---:|---:|---:|
| s6e5 F1 pit stops | 0.9508 (GBDT + MLP blend) | 0.95528 (100 models) | 0.954426 RealMLP, 5 seeds/fold | 0.0045 |
| s6e8 smartphone addiction | 0.96609 (`bag(xgbt×6)`) | 0.97101 (102 models) | 0.970706 RealMLP "repair20" | 0.0049 |
| s6e10 airline satisfaction | 0.962268 (pr60: 364 members + public pool) | 0.962836 (27 own models) | 0.962452 GPT-2 label-anywhere | 0.00057 |

All three use the same 5-fold `StratifiedKFold(shuffle=True, random_state=42)` split. **In every one of them his best
single model beats our best ensemble.** The late-campaign increments we spent days pre-registering were ±0.00001–0.00003;
the gap was 20–500× that, and in a different hypothesis class (how the data was made), not in the stacker or the
measurement.

## 2. Learnings, by area

### A. Model the data-generating process first (the biggest lever, every month)

1. **Two generators to reverse-engineer, not one.** The Playground rows come from a synthesizer (GReaT GPT-2 in s6e9/s6e10)
   fitted to an *original* dataset, and many original datasets were themselves produced by a script with a label rule.
   Deotte recovered both:
   - s6e9: the original data's purchase rule, `p_rule = Phi((s - 5.6092) / 0.8345)` with
     `s = 1.2 income/1e5 + 0.6 env + 2 subsidy - 1[anxiety Medium] - 3[anxiety High]`, used as an XGBoost base margin,
     as the target of residual encodings (`y - p_rule`), and as an offset in MLPs.
   - s6e8: the label-rule region of the original (`high if daily > 8 or social > 4`, `low if daily <= 6 and social <= 4`)
     as a feature; plus the generator's **repair** of implausible screen budgets (complete rows pile up at
     `(social + gaming + work) / daily` in [0.85, 0.95)), turned into 20 "repair posterior" channels.
   - s6e3: arithmetic identities of the original (`TotalCharges - tenure * MonthlyCharges`).

   **vs us:** our s6e8 notes concluded "the label is a noisy sigmoid of a latent index, not a hard rule" from the
   *synthetic* rows only, and our s6e5 and s6e8 notes never mention the original dataset at all. We found the budget
   identity in s6e8 (sum of parts <= daily) but stopped at a lower-bound feature.
2. **Use the original data in every model, four ways:** (a) append original rows to each fold's training side, often
   with a tuned weight (s6e5: 0.35, 0.8, 1.15); (b) priors / smoothed LLRs per exact value, per quantile bin and per
   column pair, fitted on the original only (s6e8: 583 label-free "fixed-original" columns); (c) **snap features** — each
   synthetic value's nearest original value and the difference; (d) per-class geometry — ECDF of the value among
   class-0 / class-1 original rows, distance to the nearest original of each class.
3. **GReaT specifics beyond our lesson 14:**
   - Replicate the generator *exactly*: be-great 0.0.3 + transformers 4.26.1 in a separate env; several replicas
     (55 / 60 / 65 / 70 epochs and an "organizer-default Trainer path" with data seed None) because the exact schedule
     is unknown and each replica is a different, useful view.
   - The generator was trained on a **noised copy** of the original: Deotte's reconstruction has ~7% label flips and
     ±1 jitter on 5% of ratings. He sampled 1M-row labelled **replica worlds** from such noised replicas and
     pretrained field transformers / BatchEnsemble MLPs on them before fold fine-tuning.
   - **Label anywhere** training (label at a random position, as GReaT wrote training text) gives one model that can
     be read target-first *and* target-last; it was his best s6e10 single model (0.962452, orders pooled by the mean of
     per-order LLRs).
   - Chain readout for label-anywhere models (s6e9): score `log p(label | fields before) + log p(fields after | label)`
     at every label position and average.
   - Frozen-replica **likelihood features** (per-field pseudo-likelihoods, conditional LLRs, inverse query
     `..., target is` -> next token) fed to LR / XGBoost: weaker solo (0.94) but cheap and different.
   - **BPE features:** how the generator *spelled* a number. The leading GPT-2 BPE piece of `" 92887.0"` groups values
     the way the generator saw them; target-encoding it added about +0.0002 to three different XGBoost recipes (s6e9).
   - **Temperature channels:** sampling at T = 0.7 sharpens frequencies, so log(synthetic count) vs log(original count)
     of each exact value follows a power law; the residuals are features (s6e8).
   - Engineering: tokenize each `"col is value"` fragment once per distinct value and concatenate (assert it equals
     full-text tokenization on random rows); fp16 transformer body but fp32 LM head + log-softmax for summed
     log-likelihoods; EMA weights (0.9999) for medium/large.
   - A "gentle BCE" child (one more epoch, generative loss + 0.1 × BCE on the model's own Bayes readout, softmax
     restricted to the ~1k tokens that occur) and a GPT-Neo 125M variant add diversity at the stack.
4. **Synthetic values are a finite vocabulary.** Give each numeric an exact-value categorical token (its own embedding
   per value) next to the numeric value. Digit / decimal / modulus features ("artifact columns") and radix keys
   (one integer for a numeric × categorical value) recur in s6e3, s6e4, s6e5, s6e9.
5. **Transductive, label-free density models over train + test (+ original)** are legal and strong: masked-LM /
   masked-lattice density transformers over exact-value tokens, used for completion of missing cells as a *whole
   conditional distribution* (expectation, sd, entropy integrated against a target-rate table), surprisal and entropy
   channels, and reconstruction of every field from the others (XGBoost + masked autoencoder, cross-fitted). This is the
   s6e8 chain that took RealMLP from 0.97006 to 0.97071. **vs us:** our s6e8 mask-mechanism pre-registration
   (correctly) closed mask *repair*; it did not test distributional completion, which is where his gain was.

### B. Features and encodings

6. **Nested target encoding is the default, and at scale:** hundreds of channels (s6e9: 203 — exact values, 71 floor
   binnings, log bins, smoothing 1 and 20 stored as logits; pairs, triples, n-grams), statistics beyond the mean, and
   inner `StratifiedKFold(5)` for the outer-train rows. Two anti-overfit tricks: Gaussian noise (sd 0.1 logits) on the
   training rows' TE channels, and a "pool line" (split outer-train 50/50, encode each half from the other).
7. **Pair lift**: residual of a value pair's target count against the log-odds-independence baseline of its marginals,
   shrunk (s = 20). PMI of observed pairs is the label-free counterpart.
8. **Hierarchical TE**: shrink an exact-value encoding toward its coarser group (BPE piece, `// 1000`) instead of toward
   the global prior.

### C. Models

9. **RealMLP (pytabkit `RealMLP_TD`) is the strongest single model whenever the features are right** (best single in
   s6e5 and s6e8). Settings that recur: PBLRD embeddings, Mish, small dropout, label smoothing 0.005, 32 internal
   members, epoch chosen on a 15% inner split then **refit on all outer-train rows**, 3–5 seeds per fold.
   **vs us:** s6e8 `mlp_s4` 0.942 (weight 0) and s6e5 "MLP far weaker" were implementation verdicts, not family verdicts.
10. **Base margins everywhere.** XGBoost `base_margin` / LightGBM `init_score` / CatBoost baseline from an in-fold LR,
    the recovered rule, a TE-WOE sum or a completion margin, so trees learn a correction. Most of his best GBDTs use one.
11. **Tabular foundation models: zero-shot, full context, raw features, `n_estimators=8`.** s6e10: TabPFN-3.5 0.9611 and
    Kumo-Tabular large 0.9600 beat tuned XGBoost (0.9599) and RealMLP (0.9593) with no FE. Nine TFMs went into the
    winning stack. Cost: 1–23 A100-hours each, 20–84 GB peak. **vs us:** we were blocked by licence/token handling and a
    6 GB GPU; our PR-27 independently found raw > engineered for TabPFN.
12. **Distillation:** an XGBoost teacher's soft labels on the *query* rows (held-out fold + test) plus hard labels on the
    fit rows, into MLPs or a GPT-2 Yes/No head (s6e9, 0.9464 — level with the best XGBoost, different errors).
13. **Contrast models** (s6e9): train the same net with a feature block real vs zeroed, same seeds; the scaled logit
    difference is a residual carrier for the stacker. Weak solo (0.93–0.94), earns weight anyway.
14. **Breadth of model *classes*, best of each.** His 100-model subsets ("every class, best model of each") score the
    same private LB as the full 219–468-model submissions. Classes include YDF, DCNv2, NODE, EBM, RuleFit, gcForest,
    FMs, cuML SVC / RF / kNN / UMAP-LR, TALENT models. Consistent with our lesson 7: count does not matter, kind does.
15. **Public notebooks re-run on our folds** (s6e8: 25 of them) rather than only downloading shared OOF pairs. In s6e10
    his 27 own models with no public pairs would have won alone; adding public pairs moved private LB by +0.00004.

### D. Training protocol

16. **Early stopping on an inner 10% holdout of outer-train, then refit on all outer-train rows for ~1.1× the rounds**
    (or median best iteration × 1.1, fixed for all seeds). The validation fold is never used for stopping, so OOF is
    honest. Same for NN epochs.
17. **Test prediction = mean of the 5 fold models** (no full refit); seeds averaged in logit space.
18. **Resumable per-fold parts:** each fold writes `parts/<pair>_f<k>_{valid,test}.npy`; `--folds 0,1` runs a subset,
    finished folds are skipped, and `assemble()` writes the pair when all 5 exist. Folds run in parallel on different
    GPUs or machines.

### E. Stacking

19. One **L2 logistic regression on logits** of every pair found by glob (`models/*/preds/*_oof.npy`). No Nelder-Mead,
    no member selection. s6e3: C = 0.01 with **fold-wise rank calibration** (rank each fold's OOF onto a common grid and
    replace each value by the pooled mean probability of its rank, so the 5 fold models share one scale). s6e10: C = 1
    on raw logits; values outside [0, 1] are taken as LLRs/logits as-is. cuML on GPU, refit 5–10× and averaged because
    the GPU solver is not deterministic.
20. **Metric-aware stacker for balanced accuracy** (s6e4): multinomial LR in PyTorch on per-class logits with
    balanced class weights, 5 folds × 5 seeds; two stacks blended as `argmax(P_A + 0.25 P_B)`.
21. s6e3 used a second stacking level (17 level-2 models on OOFs) but the 1-level 100-model stack was within 0.00003.

### F. Process and agents

22. **Several frontier models from different vendors, in relay** (S6E9: GPT 5.6 Sol, GPT 6 Astra, Fable 5, Fable 5.1,
    Opus 5.5; S6E10: Opus 5.5 alone). The blog's S6E3 numbers: 3 agents, ~600k lines, 850 experiments, 150 stacked.
23. **Keep every experiment's OOF/test pair, good or bad**, and let the L2 stacker decide. Selection happens at the stack
    by regularisation, not by a per-leg acceptance bar.
24. **Agent prompts that recur in the blog:** "summarise all experiments", "write one single model stronger than all of
    these", "distil all OOF/test predictions into one model". An agent also rewrote TFM libraries for full context,
    mixed precision and KV-cache inference, with pilot runs verifying parity with the original accuracy.
25. **Compute is part of the method:** A100/80 GB-class GPUs, GPU everything (cuDF, cuML, GPU GBDTs). Our 6 GB card
    rules out the GPT-2 medium/large legs, full-context TFMs and the density transformers locally.

## 3. Where our approach was right, and where it cost us

Confirmed by his results: the shared 5-fold seed-42 split; LR-on-logits stacking with strong L2 (our pr42 landed on
C 0.01 independently); keyed OOF/test pairs; fold-safe TE; "differs in kind" (lesson 7); raw features for TFMs.

Cost us: effort allocation. From s6e10 PR-43 to PR-60 we measured ±0.00001 levers with care while a 0.0006 lever
(the generator) was untried until too late, and in s6e5/s6e8 a 0.0045–0.0049 lever (original data + rules + RealMLP
done right) was never on the list. Pre-registration protects against fooling ourselves about small effects; it does not
tell us which large effects we have not looked for.

## 4. Suggested repo changes (not done in this commit)

1. `kaggle.original`: snap features, per-key / per-pair smoothed LLRs from the original, per-class ECDF/nearest
   distance, `append_original(weight=...)` for fold training sides.
2. `kaggle.generator`: GReaT row text + fragment tokenization with the exactness check, BPE-piece table, temperature
   power-law channels, label-first / label-anywhere scoring helpers. Pinned replica env documented.
3. `run_cv`: `base_margin=` hook; inner-holdout early stopping + refit-at-1.1× mode; per-fold `parts/` with resume.
4. `stack_oof`: fold-wise rank calibration option; accept logit/LLR inputs without re-logit.
5. `new-competition` skill: a day-1 checklist — download the original dataset, test for its label rule and arithmetic
   identities, check the GReaT tokenization signature, launch a TFM zero-shot leg and a pytabkit RealMLP baseline.
