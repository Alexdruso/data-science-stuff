# Predicting Credit Card Fraud (playground-series-s6e11)

**Task**: binary classification — predict `is_fraud` (1.67% positive)
**Metric**: ROC AUC (sample submission is a probability)
**Deadline**: 2026-11-30 23:59 UTC — queried from the source, `kaggle competitions list -s s6e11`,
on 2026-11-01.

---

## ⚠️ CRITICAL: Row-order invariant — always load labels and IDs through build_features()

`features.py::build_features()` sorts every dataframe by `["id"]`. Predictions are saved as keyed
parquet (`results/oof_<name>.parquet` with `id`, `fold`, `pred`) via
`data_science_stuff.kaggle.predictions`; `load_predictions(results, name, ids)` aligns by id.
`y` and ids **must** still come from `build_features()`.

## ⚠️ No leakage — fold-aware rules

1. Target encoding is fold-aware, never global.
2. Original-data aggregates (orig target mean per value) use only the 20k original rows — no
   train targets — so they are leak-free.
3. Note: `cv.fold_assignment` returns folds **1..5**, not 0..4.

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | 699,960 | id + 24 features + `is_fraud` |
| test | 299,984 | id + 24 features |
| original | 20,000 | `data/orig/credit_card_fraud_2026.csv` (Kaggle `uditjain13/credit-card-fraud-detection-2026`), fraud rate 1.70% |

13 numeric (`amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`,
`card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`,
`time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes`), 5 categoricals
(`merchant_category` 12, `card_type` 5, `auth_method` 5, `channel` 5, `device_type` 8), 6 booleans.
No nulls anywhere.

## EDA findings (Day 1, 2026-11-01)

- The original dataset is the generator source: ~97-100% of train values for amount /
  velocity / merchant_risk occur in it; train ranges are clipped inside orig ranges.
- **Signal is close to additive/logistic**: plain LR on one-hot + standardized numerics gives
  OOF 0.9418 (splines 0.9434) vs XGB 0.9468. On the original data alone, LR CV = 0.943.
  Shallow trees beat deep ones (XGB d3 0.94758 > d4 0.94726 > d6 0.94675).
- Strongest drivers (orig LR, standardized): merchant_risk_score, velocity_score,
  cvv_retry_count, is_new_merchant, ip_country_mismatch, auth_method=No Authentication,
  billing_shipping_mismatch, used_vpn, is_ai_generated_scam_attempt.
- No hidden-ID numerics: per-column target encoding never beats the raw monotone AUC.
- Appending the original rows to training folds: flat (+0.00001, XGB d6).

## Current best (2026-11-03 17:30)

- **stack_v5** = LR-on-logits(ebm_i30_s7, ebm_i30_mib256, ebm_i30_s7_mib256, ebm_i30, ebm_i60,
  xgbt2_b2048, xgbt2_b2048_s1, lgbm_d2_b2048, **lm_distilgpt2_ep10**), OOF **0.95085** —
  submitted. LM ep10 solo 0.94987, stack weight .65. Earlier: stack_v4 (ep5 LM) 0.95024,
  stack_v3 (2-ep LM) 0.94920.

**Best public: stack_ext1** (public goodpjw digit-TE OOF bank + our LM ep10, `src/ext_stack.py`),
OOF 0.95194, **LB 0.95212** (top 0.95229 on 11-03). See "11-03 evening" below.

**Next (in flight at hand-off 11-03 ~21:00):**
- Local GPU queue: `lm_distilgpt2_ep10_lr5` fold-1 pilot (`logs/lm_ep10_lr5_f1.log`) →
  column-order rescore K=1..16 of the ep10 f1 model (`logs/lm_rescore_ep10_f1.log`) →
  GReaT-format LM pilot (`--great_format`: full names, "name is value,", label "is_fraud is y,";
  bs 32, `logs/lm_great_ep5_f1.log`). Read each with `src/lm_fold_eval.py` **and** in the
  ext stack (folds-subset snippet logged 11-03), since gains inside our own stack did not
  transfer (ep20: +0.0002 in ours, 0 in ext).
- Kaggle `s6e11-lm-ep20b` (ep20 folds 4-5) finishes ~23:00; low priority now (ep20 adds 0 in the
  ext stack). Kaggle weekly GPU quota is nearly used (~25 h of 30).
- Levers: more LM *kinds* (format, tokenisation, label-first variants) priced inside the ext
  stack; re-pull public OOF banks every day (new members appear); our own digit-TE trees are
  redundant with the bank unless we add something new to them.

## Previous best (2026-11-01 21:25)

- **stack_v2** = LR-on-logits(ebm_i30, ebm_i60, xgbt2_b2048 ×4 seeds, lgbm_d2_b2048), OOF
  **0.94843** — submitted. stack_v1 (no i60/lgbm) 0.94842 also submitted. Best single: ebm_i60
  0.94827 / ebm_i30 0.94824. Submissions so far: xgb_s42, xgbt2_s42, stack_v1, stack_v2.
- Running at hand-off: `ebm_i30_s7` (seed-7 EBM, procs logs/q17.log folds 1-3, q18.log folds
  4-5; the last process assembles `oof_ebm_i30_s7.parquet`; re-run
  `python src/ebm_leg.py --inter 30 --seed 7` to assemble if needed). Add it to the stack.

## Experiments log

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-01 | xgb_s42 (d6) | 0.94675 | baseline, submitted |
| 11-01 | xgb_orig_s42 | 0.94676 | + orig rows: flat |
| 11-01 | xgb_d4 | 0.94726 | |
| 11-01 | xgb_d3 | 0.94758 | shallower is better |
| 11-01 | xgb_d2 (lr .05, col .8) | **0.94763** | best single so far |
| 11-01 | xgb_d1 | 0.94710 | pure additive is short: pairwise interactions matter |
| 11-01 | xgb_d3_fe (value counts in train+test & orig, amt/bal) | killed | fold1 −0.00035, fold2 −0.00066 vs d3 |
| 11-01 | lrspl_k10 (quantile splines + one-hot) | 0.94361 | stack weight ≈0 over XGB legs |
| 11-01 | stack LR(xgb d2,d3,d4,d6) | 0.94770 | |
| 11-01 | cat d6 lr .06 | fold1 0.94432 | deep trees again weaker |

- "Value absent from orig" flags carry little signal (amount: 2.4% vs 1.65% fraud on 2.6% of
  rows); no exact orig rows in train.
| 11-01 | tune_xgb (40 TPE trials, folds 1-2) | 0.94799 (f1-2) | best d2, gamma .73, max_bin 512, colsample ~1, lambda .03 → `results/best_params_xgb.json` |
| 11-01 | xgbt2_s42 (tuned d2) | **0.94788** | submitted |
| 11-01 | xgbt3_s42 (best d3 trial) | 0.94774 | |
| 11-01 | cat d6 | killed after 3 folds | ~−0.0015 vs XGB per fold; CPU better spent on EBM |
| 11-01 | realmlp 48 ep bs1024 | f1 0.93640, f2 0.93583 | weak NN (256-ep default would take 2.5 h) |
| 11-01 | **ebm_i30** (interpret EBM, 30 pairs, 4 bags) | f1 **0.94688** | best single on fold 1 (xgbt2 f1 0.94624); ~30 min/fold at 2 jobs |
| 11-01 | stack LR(xgbt2,d2,d3,d4,lrspl) | 0.94787 | |

- **GA2M structure**: d2 > d1 and EBM (additive + pairwise) > every tree model → the label
  generator is additive + a few pairwise interactions. Lever direction: better shape functions
  (EBM more bags / interactions, smoothing), not deeper trees.
| 11-01 | xgbt2 seeds 1,2 (max_bin 512) | 0.94790 / 0.94782 | |
| 11-01 | **xgbt2_b2048** (tuned d2, max_bin 2048) | **0.94807** | +0.0002 over 512, all 5 folds up; 8192 identical (0.94807) → saturated |
| 11-01 | xgbt2_b2048 s1 / s2 | 0.94812 / 0.94805 | |
| 11-01 | realmlp_s42 (48 ep) | 0.93243 | weak; stack weight ≈ −0.06 |
| 11-01 | ebm_i30 folds 1,2,4 | 0.94688 / 0.94955 / 0.94782 | vs b2048 0.94652 / 0.94991 / 0.94804 → level, different kind |
| 11-01 | stack LR(b2048×3, xgbt2, d3, lrspl, realmlp) | 0.94810 | before EBM |

- **Fine binning matters**: max_bin 256→512→2048 kept improving XGB d2 (tuner picked 512 from
  {64..512}); the shape functions have sharp local structure (possibly values copied from orig).
| 11-01 | xgbt2_b2048 s3 / lr 0.01 | 0.94810 / 0.94804 | lr 0.01 no gain |
| 11-01 | per-value residual TE (fold-honest, every numeric) on xgbt2_b2048 | +0.00000 | no exact-value memorisation signal; fine-bin gain is threshold precision |
| 11-01 | **ebm_i30** (5 folds) | **0.94824** | best single; folds .94688/.94955/.95278/.94782/.94430 |
| 11-01 | **stack_v1** LR(ebm_i30, b2048×4) | **0.94842** | EBM weight .67; submitted. Adding lr01/lrspl/realmlp: same 0.94842 |
| 11-01 | lgbm_d2_b2048 / xgbt3_b2048 | 0.94802 / 0.94787 | stack with them: 0.94842 (flat) → tree legs are absorbed |

**Lever direction (end of Day 1):** EBM carries the stack (weight ~.67); every extra tree leg is
absorbed. Next: EBM variants (more interactions `ebm_i60` running, more outer bags, other seeds,
`smoothing_rounds`, `max_interaction_bins`), and EBM on more cores. A GReaT label-first GPT-2
leg (root lesson 14) is the untested different-kind lever — needs cloud GPU.
| 11-01 | ebm_i60 | 0.94827 | folds vs i30: −.00011/+.00009/+.00003/+.00002/+.00014 |
| 11-01 | xgbt2_b2048_orig (+20k orig rows) | 0.94793 | −0.00014 vs 0.94807 → orig rows hurt at d2 |
| 11-01 | xgbt2_b2048_oh (max_cat_to_onehot 16) | 0.94798 | −0.00009 → keep partition splits |
| 11-01 | xgbt2_b2048_spw5 (scale_pos_weight 5) | 0.94790 | −0.00017 |
| 11-01 | ebm_i30_s7 (seed 7) | f1 0.94706, f4 0.94793 | (i30 .94688/.94782); folds 2,3,5 finish after 22:00 → assemble + restack next session |
| 11-01 | **stack_v2** (+ebm_i60, lgbm) | **0.94843** | EBM weights .34/.33; submitted. Stacker C 0.01-100 insensitive |

**Next session:** (1) assemble/stack `ebm_i30_s7`; (2) EBM knobs not yet tried — outer_bags 8+,
`max_interaction_bins`, `smoothing_rounds`, `learning_rate`; give it all cores, one heavy job at
a time; (3) day-1 GReaT fingerprint check (root lesson 14) and plan a cloud-GPU label-first
GPT-2 leg — 29 days remain.

## Day 2 (2026-11-02)

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-02 | ebm_i30_s7 assembled | **0.94837** | best single; stack (3 EBMs + 4 b2048 + lgbm) 0.94847 (+0.00004 ≈ noise) |
| 11-02 | ebm_i30_mib256 | 0.94832 | +0.00008 vs i30, every fold ≥ i30 |
| 11-02 | ebm_i30_mib128_sm150 | 0.94827 | flat; stack of 5 EBMs + 5 trees 0.94848 → EBM side saturated |
| 11-02 | ebm_i30_mb4096 | f1 0.94659 | killed — finer main bins don't help EBM |
| 11-02 | **lm_distilgpt2 pilot** (label-first GPT-2 Storyteller, `src/lm_leg.py`, 2 ep, 25% negs) | sub 0.94331 | fold-1 30k-row subset: EBM 0.94334, base3 stack 0.94343 → **+LM 0.94479 (+0.0014)**; corr(LM, EBM) 0.936 → different kind. `src/lm_sub_eval.py` |
| 11-02 | **lm_distilgpt2** fold 1 (full 140k val, K=3) | f1 0.94536 | below EBM f1 .947 solo, but fold-1 LR stack of 6 legs 0.94711 → **+LM 0.94780 (+0.0007)**, LM weight .39 (largest) |
| 11-02 | **lm_distilgpt2** (5 folds) | 0.94657 | folds .94536/.94776/.95173/.94678/.94167; corr with EBM 0.925 |
| 11-02 | ebm_i30_s7_mib256 (8 bags) | **0.94846** | best single EBM |
| 11-02 | **stack_v3** LR(5 EBMs, 3 trees, lm_distilgpt2) | **0.94920** | same legs without LM 0.94851 → **+0.00069**; LM weight .39; submitted |
| 11-02 | lm_distilgpt2_orig10 (replica: 10 ep on 20k orig rows, then 2 ep/fold) | f1 0.94550 | fold-1 stack gain +0.00074 vs +0.00071 plain → orig pre-fit ≈ noise; both LMs together +0.00076 (a 2nd LM run adds little). `src/lm_fold_eval.py` |
| 11-02 | **lm_distilgpt2_ep5** (5 epochs instead of 2) | f1 **0.94733** | solo ≈ best EBM on fold 1 (.94706); fold-1 stack gain **+0.00159** (2 ep: +0.00071) → **training length is the lever**; loss 1.1395 still falling. Full 5-fold run started 17:50 (~80 min/fold, ends ~23:45) |
| 11-02 | lm_distilgpt2_ep5 fold 2 | f2 0.94963 | ≈ EBM f2 (.94955); fold-2 stack gain **+0.00117** (2 ep: +0.00060) → replicated |
| 11-02 | lm_distilgpt2_ep5 fold 3 | f3 **0.95421** | **beats EBM f3 (.95278) solo** — LM is now the best single leg; fold-3 stack gain **+0.00192** (2 ep +0.00076) |
| 11-02 | lm_distilgpt2_ep5 fold 4 | f4 **0.95000** | beats EBM f4 (.94808) by .0019; fold-4 stack gain **+0.00258** (2 ep +0.00076) |

- **LM leg mechanics** (`src/lm_leg.py`, overlay venv `~/venvs/lm` = main venv + transformers):
  rows as `" short-name value,"` segments (CSV text formatting kept, names shortened to halve
  tokens), label first (`fraud 1,`), column order shuffled per step; softmax restricted to the
  ~used tokens (full 50k-vocab logits OOM on 6 GB). distilgpt2 fp16 trains ~245 rows/s
  (~10 min/epoch at 147k rows), scores ~800 seq/s. Score = logsumexp over K orders of
  log P(x|1) − log P(x|0) + prior. Full run: 2 ep, K=3 val+test, ~70 min/fold.

- **GReaT fingerprint (partial, 11-01):** train values copy orig's text formatting exactly
  (e.g. amount: 92% 2-decimal / 8% 1-decimal, i.e. trailing zero dropped, in train *and*
  orig; velocity/merchant_risk 1 decimal everywhere; one 2-decimal merchant_risk in 700k).
  Consistent with a text LM generator. The GPT-2 tokenizer / label-first checks are still
  TODO (`transformers` is not in the venv).

## Day 3 (2026-11-03)

- The 11-02 23:08 shutdown hit fold 4 during test scoring: `f4_val.npy` existed but no model /
  test preds, and the resume logic would have skipped the fold. Fixed (`lm_leg.py` now saves the
  model *before* scoring); fold 4 is retrained (orphan val kept as `f4_val_orphan.npy`).
- `lm_leg.py --label_last` adds the Detective (label-last, P(y|x) read at `" fraud"`, mean
  logit over K orders). `src/kaggle_lm_build.py` packages `lm_leg.py` as a private Kaggle
  T4×2 kernel running one job per GPU (outputs → `results/_<name>/`, assemble locally).

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-03 | meta-learner over stack_v3 legs (`src/meta_eval.py`) | LR 0.94921 / LGBM d2 0.94873 / LGBM d2 + raw 0.94875 | GBDT meta loses 0.0005; raw features add nothing → keep LR |
| 11-03 | lm_distilgpt2_ep5 fold 4 retrained | f4 0.94915 | lost run gave .95000 (GPU nondeterminism); fold gains f1-f4 +.00159/.00117/.00192/.00202; adding the 2-ep LM on top: +.00003 avg → drop it |
| 11-03 | **lm_distilgpt2_ep10** fold 1 (Kaggle T4) | f1 **0.94794** | fold-1 stack gain **+0.00199** (ep5 +0.00159, ep2 +0.00071) → still climbing; full run on Kaggle (folds 2-5, `s6e11-lm-ep10`), fold 1 locally |
| 11-03 | lm_distilgpt2_last_ep5 (Detective, label-last) fold 1 | f1 0.94317 | stack gain **+0.00001** → absorbed; dead |
| 11-03 | **lm_distilgpt2_ep5** (5 folds) | **0.94904** | best single leg; f5 .94513 (stack gain +.00192) |
| 11-03 | **stack_v4** (stack_v3 legs, LM 2ep → ep5) | **0.95024** | +0.00104 over v3; submitted |
| 11-03 | lm_distilgpt2_ep10 fold 1 (local) | f1 **0.94846** | stack gain **+0.00230** (Kaggle pilot run of the same config +0.00199: seed noise ~.0003); averaging the two runs (corr .9925) +0.00225 → not variance, capacity/length is the lever |
| 11-03 | stack C 0.01-10, or 3 legs (best EBM + xgb + LM) | 0.95024-0.95025 | stack is just EBM + LM; only a better LM moves it |
| 11-03 | lm_gpt2_ep5 (124M) fold 1 (Kaggle) | f1 0.94756 | stack gain +0.00166 ≈ distilgpt2 ep5 (+0.00159) at 2× cost → capacity is not the lever |
| 11-03 | lm_distilgpt2_ep20 fold 1 (Kaggle) | f1 0.94745 | solo below ep10 but stack gain **+0.00254** (ep10 +0.00230); ep10 + ep20 + gpt2 together +0.00269 → full ep20 run (Kaggle `s6e11-lm-ep20a` folds 2-3, later folds 4-5; fold 1 local) |
| 11-03 | lm_distilgpt2_ep5_lr5 (lr 5e-4) fold 1 | f1 0.94793 | stack gain +0.00199 (lr 2e-4: +.00159; ep10: +.00199/+.00230) → higher lr substitutes ~half the epochs |
| 11-03 | **lm_distilgpt2_ep10** (5 folds; f2-5 Kaggle T4, f1 local) | **0.94987** | folds .94846/.94953/.95518/.95013/.94611 — every fold > ep5. CPU assembly: `src/lm_assemble.py` |
| 11-03 | **stack_v5** (stack_v4 legs, LM ep5 → ep10) | **0.95085** | +0.00061 over v4; + ep5 as an extra leg: 0.95085 (flat); submitted |
| 11-03 | nonlinear stack terms (LM², EBM², LM×EBM, LM hinge splines) | 0.95081-0.95085 | flat vs linear 0.95085 |
| 11-03 | lm_distilgpt2_ep20 fold 1 (local) | f1 0.94750 | stack gain +0.00237 (ep10 +0.00230); ep10 + ep20 together +0.00254 (corr .978) → ~+0.0002 expected from adding ep20 |

### 11-03 evening — we were behind the public notebooks (LB 0.95110 vs top 0.95229)

- **Missed lever: digit fingerprint.** Public notebooks target-encode *digit keys* of the six
  float columns (int part, frac, int/10, int(10x), leading 1-2 digits + length, GPT-2 tokens:
  "96-key digit-trie TE", nested inner-fold TE, smoothing 20/100) → a single XGB d2 / LGBM
  4-leaf reaches **0.9505-0.9515 OOF** (our best tree 0.94807, EBM 0.94846). Fraud rate depends on
  the first decimal (.0/.4 high, .2/.3/.6 low) in every float column. Sources:
  `lazyyeou/s6e11-digit-trie-te-joint-glm-nested-stack`, `goodpjw2008/s6e11-part-6-glm-boost-stack-lb-0-95215`.
- **Public OOF banks share our folds** (StratifiedKFold(5, True, 42), train.csv is id-sorted):
  `goodpjw2008/s6e11-fraud-series-artifacts` → `ext/goodpjw/` (26 members, best gb_bag5 0.95151),
  `nilanjansaha123/96-key-digit-trie-gpt-2-full-refit-hybrid` output → `ext/96-key-…/preds/`.
  `src/ext_stack.py` = LR on logit-ranks over ext + our legs.

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-03 | ext bank alone (goodpjw) | 0.95159 | |
| 11-03 | **stack_ext1** = goodpjw bank + our ebm×2, xgbt2_b2048, lm_distilgpt2_ep10 | **0.95194** | LM weight .50 (largest) → LM is the different-kind leg; **LB 0.95212** (v5 0.95110); submitted |
| 11-03 | + 96-key bank (5 members) | 0.95192 | flat — same kind |
| 11-03 | lm_distilgpt2_ep20 folds 2, 3 (Kaggle) | .94832 / .95334 | solo below ep10 (.94953/.95518) |
| 11-03 | ext stack, folds 1-3 only, members rank-logit within subset | base .95314 | ext bank alone −0.00028; ext + LM ep10 alone = base (our EBM/tree legs add 0 over the bank); **+ ep20: −0.00002, ep10→ep20 −0.00007** → ep20 does not help in the ext stack; LM length saturated at ep10 there |

## Day 4 (2026-11-04)

- LB top 0.95240 (09:00). New public: masayakawamata GPT-2 label-first OOF bank (GReaT recipe,
  gpt2 small, 10 orders, CV **0.95003**; dataset `masayakawamata/s6e11-gpt2-target-first-oof` →
  `ext/dl/`), goodpjw Part 7 (residual MLPs from the GLM `gbn_bag5_w*`, + masa GPT-2, nested CV
  0.95206, LB 0.95222). Downloads land in `ext/dl/` (alexchilton, malyshevdanil banks slow).
- `src/pool_stack.py` = general bank loader (+ masa `oof.npy/test.npy`, `oof_*.npy` formats),
  dedup by rank corr, LR stack with nested outer CV, `--ablate`.

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-04 | lm_distilgpt2_ep20 (5 folds, f4/f5 Kaggle ep20b) | 0.94862 | below ep10 |
| 11-04 | pool: goodpjw artifacts + 96-key + our ep10/ebm | 0.951911 | ≈ stack_ext1 |
| 11-04 | **pool_v1** = above + masa GPT-2 + our ep10/ep20/ebm×2/xgb | **0.952033** | without ours 0.951985 → **our LMs add only +0.00005 once masa's LM is in**; masa alone adds +0.00035. C 0.01-10 flat |
| 11-04 | masa vs ep10 per fold | masa .94888/.95032/.95531/.94987/.94585 vs ep10 .94846/.94953/.95518/.95013/.94611 | rank corr .98; avg of the two +0.0005 solo but +0.00005 in the stack → same-kind LMs are absorbed |
| 11-04 | residual LightGBM (4 leaves) on pool_v1 logit, raw + digit features | 0.952017 | null: the pool already has what shallow trees see |
| 11-04 | GReaT-format pilot | killed | masa's LM already *is* GReaT format → same kind, absorbed |
| 11-04 | per-column LM LLRs (ep10 f1, K=4; `src/lm_colscore.py`, `lm_leg.py --colscore`) | f1 gain −0.0003 | 24 col-LLR features on top of a single pooled-stack logit lose; LM total adds −0.00002 → dead |
| 11-04 | orig-row copy check (best orig match over 6 high-card cols) | — | max 3/6 matches, mostly 1 → the generator does not copy orig rows; dead |
| 11-04 | **pool_v2** = all banks (goodpjw, 96-key, masa, alexchilton 38, malyshev own 11) + ours | **0.952188** | alexchilton +0.00014, malyshev +0.00003, ours +0.00008; **submitted** |
| 11-04 | residual LightGBM on pool_v2 with raw + LM logits | 0.952085 | nonlinear meta loses 0.0001 → keep LR |
| 11-04 | fold-1 LM gains over pool_noLM logit (`src/lm_pool_eval.py`) | ep5 −.00004, ep10 +.00000, ep20 +.00001, lr5 −.00007 | the pool has absorbed the LM kind; only a much stronger LM can move it |

**PR (pre-registered 11-04 ~10:40, before the number):** `lm_distilgpt2_allneg` = all negatives
(neg_frac 1.0) + positives ×4, 2.5 epochs (same steps as ep10), fold 1, K=3. Adopt (full 5-fold
leg: Kaggle f2/f3 + local f1/f4/f5) only if f1 solo ≥ 0.94896 (ep10 0.94846 + 0.0005) **and** its
fold-1 gain over pool_noLM beats ep10's (+0.000004). Honest expectation: solo +0.0002-0.0005,
gain within noise → likely fail.
| 11-04 | K=16 vs K=3 column orders (ep10 f1, 40k rows) inside pool_noLM | gain −0.00003/−0.00002 | solo +0.0001 but null in the pool → more orders is dead |
| 11-04 | adversarial validation train vs test (raw, 200k each) | AUC 0.5004 | no shift → OOF ordering trustworthy |
| 11-04 | pool + kernel outputs (sameershaikh glm-boost bag, megayak per-value TE, yekenot RealMLP ×2 csv→`ext/ko_conv`) | 0.952251 (C .1) | |
| 11-04 | pool sweep: C .01/.03 | 0.952289/.952287 | stronger reg helps the bigger pool; pruning (dedup .999/.998, min_auc .948/.949) **loses** (.95222/.95221/.95220/.95205) → weak members carry signal |
| 11-04 | **pool_v3** = 105 members (min_auc .93, dedup .9999, all our legs, C .01) | **0.952304** | ours +0.00010; **submitted**. C .003 0.952221 |

**PR (11-04 ~11:00, before the number): TabPFN-3 leg** (`src/run_tabpfn.py`, v3 checkpoint, ctx = all
fold positives + 90k negatives, raw + digit keys, n_est 2). Bar: fold-1 gain over pool_v3-without-it
≥ +0.00005 (fold-level single-logit read) → then full 5 folds. Prior ~15% (s6e10: our local
TabPFN-3 null; only full-context public 3.5 earned weight).
| 11-04 | raw standardized logits as stacker input (`--raw`) | 0.952230 | rank-logit 0.952304 → keep rank-logit |
| 11-04 | **pool_v4** = 142 members: + ko2 kernel outputs (sameershaikh 10-fold glm-boost bag, seed set, digit-key trees, realmlp-glm; dhananjayk part 3; lazyyeou; georgymamarin) via `ext/ko2`, `ext/ko2_conv` (csv→npy); public blends/stacks dropped (`--drop blend pooled fusion _blend avg5`) | **0.952484** | ours +0.0001 (LMs ep20/ep10/ep5 = coefficients 2-4); **submitted** |

Pool recipe (reproduce): `src/pool_stack.py --ext ext/dl/s6e11-fraud-series-artifacts
ext/96-key-digit-trie-gpt-2-full-refit-hybrid ext/dl/s6e11-gpt2-target-first-oof ext/dl/s6e11-oof-predictions
ext/dl/malyshev_split ext/ko ext/ko_conv ext/goodpjw ext/ko2 ext/ko2_conv --ours <all our legs> --min_auc 0.93
--dedup 0.9999 --C 0.01 --drop blend pooled fusion _blend avg5`. Keep harvesting kernel outputs — every
batch so far has added +0.00005-0.0002.
| 11-04 | **PR all-negatives LM** (`--neg_frac 1 --pos_rep 4 --epochs 2.5`, f1) | f1 **0.94833** | ep10 0.94846 → **FAIL** vs bar 0.94896; pool gain −0.00004. More distinct negatives at equal compute does not help |
| 11-04 | pool + ko3 batch (16 more kernels; npy members in `ext/ko3`, parquet/csv → `ext/ko3_conv`) | 0.952509 | without the 22 ko3_conv members 0.952523 → the 169-member LR starts to overfit; pool gains now ~noise (≤0.00004); ours +0.00005 |
| 11-04 | C .005/.02 on the 169-member pool | .952509/.952511 | flat |
| 11-04 | **instrument calibration**: paired bootstrap (100×, rows) of pool deltas | sd **0.00004** | v4−v3 +0.00018 (4.6 sd), v3−v2 +0.00012, v4−stack_ext1 +0.00054 (sd .0001) → pool deltas ≤0.00004 are noise |
| 11-04 | **PR TabPFN-3** (v3 ckpt, ctx all pos + 90k neg, raw + digit keys, n_est 2) fold 1 | f1 solo **0.94311** | rho(masa) .67 but pool_v4 fold-1 gain −0.000001 → **FAIL** (bar +0.00005); 20 min/fold val. Dead, as in s6e10 |

**PR (11-04 12:25, before the number): semi-supervised Storyteller** (`lm_leg.py --semi
_lm_distilgpt2_ep10`): fold-k fit set += test rows included w.p. q = p + 0.25(1−p), label
re-sampled every epoch ~ Bernoulli(p/q), p = the ep10 **fold-k** model's test posterior (honest
teacher: never saw fold k). Rationale: P(x|fraud) is positive-starved (9.3k rows/fold); test holds
~5k unlabeled frauds (generative EM). f1 pilot, 8 epochs (28k steps vs ep10's 23k). Bar: f1 solo
≥ 0.94896 (ep10 + 0.0005) → full leg. Prior ~20%.
| 11-04 | pool_v5 = pool_v4 sources + ko3 npy members (no ko3_conv) | 0.952523 | +0.00004 vs v4 = 1 sd (noise) → not submitted |
| 11-04 | two-level family stack (per-family LR C .01 → LR over 5 family logits) | 0.95233 | flat LR 0.952523 wins. Family stacks: tree .95186, nn .95139, lm .95077 (5 LMs; masa alone .95003), glm .95073, ebm .95039 |
| 11-04 | scratch-init distilgpt2 LM (no pretraining), ep10, lr 5e-4 / 1e-3, f1 (Kaggle `s6e11-lm-scratch`) | f1 0.94632 / 0.94582 | rho(masa) .970/.974 (more decorrelated) but pool_v4 fold-1 gain −0.00004/+0.00001 = noise → dead. Pretraining helps solo by +0.002 |

**PR (11-04 13:50, before the number): TabPFN-3.5 on Kaggle T4×2** (user accepted the licence;
`kaggle_kernels/s6e11-tabpfn35`, raw features, ctx 150k majority-downsample = all positives,
n_est 2). Fold-1 pilot. Bar: pool_v4 fold-1 gain ≥ +0.00005 → folds 2-5. Prior ~25% (s6e10:
public full-context 3.5 was the top coefficient; our local v3 here was null).
| 11-04 | **PR TabPFN-3.5** (Kaggle T4×2, ctx 150k majority-downsample, raw, n_est 2) fold 1 | f1 solo **0.94717** | 8 min val + 7 min test per fold. pool_v4 fold-1 gain (logit input) **−0.00005 → FAIL**; TabPFN-3 re-read with logits −0.000015 (fail stands; earlier read fed probabilities — `lm_pool_eval.py` now logit-transforms probability members). Not run on folds 2-5 |
| 11-04 | **PR semi-supervised Storyteller** (`--semi _lm_distilgpt2_ep10`, +79.7k test rows, ~5.5k soft positives, ep8) f1 | f1 **0.94796** | ep10 0.94846 → **FAIL** (bar 0.94896); pool_v4 fold-1 gain −0.00002. Unlabeled test rows do not help the generative LM |
| 11-04 | ko4 harvest (areebbb pool 0.95231 = submissions only; alexchilton update; denpugovkin logreg → `ext/ko4_conv`) | 0.952523 | alexchilton update adds no new member after dedup |

### 11-04 hand-off
- **Best OOF: pool_v5 0.952523** (not submitted, +1 sd over v4); **submitted: pool_v2/v3/v4** (v4 0.952484).
- In flight: `lm_distilgpt2_great_ep10` full leg (GReaT format, bs 32, ~4 h/fold incl. scoring,
  resumable: re-run the same command, `logs/lm_great_ep10.log`):
  `HF_HUB_OFFLINE=1 ~/venvs/lm/bin/python src/lm_leg.py --model distilgpt2 --epochs 10 --great_format --bs 32 --k_val 3 --k_test 2 --tag _great_ep10`.
  Price it only in the full pool (fold-level single-logit reads cannot resolve LM variants: sd ≈ gain).
- Dead today (all pre-registered): per-column LLR, K orders, all-negatives, semi-supervised EM,
  scratch-init, TabPFN-3 / 3.5, residual boosting, family stack, raw-logit stacker, orig-copy.
- The lever that keeps paying is **harvesting public same-fold banks** (+0.0001-0.0002 per batch);
  re-list kernels by date every few hours (`ext/ko*/`, convert csv/parquet to `ext/ko*_conv`).

### 11-04 afternoon — own lever: model the generator (replica), not the samples
- Digit fingerprint is **generator-made**: in orig, per-first-decimal fraud rates are noise; in train
  ".0" is fraud-elevated in every float column (amount 1.90% vs orig 1.40%, distance 1.94/1.34,
  hours 2.20/1.84) and digit frequencies are distorted (hours ".9" 48k vs ".5" 83k) = tempered
  token sampling. → a replica of the generator scored like the sampler should reproduce it.
- Prototype memorisation probe (`src/proto_match.py`): max exact matches to the 339 orig fraud rows
  ≤ 16/24 (mostly 11-12) → no row copying; "closer to orig non-fraud" diff AUC 0.82 but pool
  +0.000036 (1 sd) → not a lever.
- `src/replica.py`: distilgpt2, GReaT text, label-first, trained on the 20k ORIGINAL rows only (no
  train labels → leak-free on train, no CV), scored on a 30k fold-1 subset (`src/replica_eval.py`):

| epochs (lr 5e-5, bs 32) | T=1 AUC | T=0.7 AUC | T=0.7 top-k 50 | mean loglik T=1 |
|---|---|---|---|---|
| 10 | 0.87087 | 0.86929 | 0.67225 | −128.457 |
| 20 | 0.91764 | 0.91761 | 0.79862 | −127.991 |
| 40 | 0.92779 | 0.92729 | 0.82488 | −127.126 (T=0.7: −125.910) |

Pool (pool_v4) gain on the 30k subset: e40 −0.00005 (T=1) / −0.00002 (T=0.7). Flattening
(+0.047, then +0.010 for 2× epochs). Train rows are consistently more likely under T=0.7 than T=1
→ data really were sampled tempered. Running e80 (resume from e40) to close it on data, not extrapolation.
| **80** | 0.91962 | 0.91922 | 0.78465 | **−126.163** (T=0.7 −124.804) |

**e80 overfits orig (solo ↓) but its pool_v4 gain jumps to +0.0006** on the 30k subset: +0.00041…
+0.00063 over 5 inner-CV seeds (e40: −0.00027…+0.00011); bootstrap mean +0.00058, sd 0.00038,
P(≤0) 7%. Coefficient **negative** (−0.20 vs pool 1.12) → a suppressor: removes a component the pool
over-trusts. Hypothesis: the generator memorised orig; only a replica that memorises too sees it.
No train labels used → no leakage, no CV.

**PR (11-04 19:35, before the number):** full fold-1 read (140k rows, `src/replica_score.py`, K=2,
T=1, fold-1 rows scored first). Bar: pool_v4 single-logit fold-1 gain ≥ +0.0002 → score all
train+test and add to the pool (full-pool delta must then be ≥ +0.0001 = 2.5 sd). Next if it passes:
longer replicas (e120/e160) — the gain rose with memorisation.
| 11-04 | **PR replica e80, full fold 1** (140k rows, K=2, T=1) | solo 0.93021 | pool_v4 gain **-0.00005**, coef -0.002 -> **FAIL**. The 30k-subset +0.0006 was noise (bootstrap P(<=0)=7%, landed in the tail). Full scoring stopped. An orig-only replica does not add to the pool at e10-e80 |
| 11-04 | pool calibration vs joint generator-artifact counts (#".0" decimals, #short numbers across 6 floats) | ratio y/pool 0.99-1.02 on all groups >18k rows | pool already calibrated along the artifact axis; dead |
| 11-04 | **pool_v5 submitted** (0.952523) | | |

**Day-4 verdict:** every different-kind leg built today (LM per-column, semi-supervised EM, scratch init, TabPFN-3/3.5, orig-only generator replica e10-e80, prototype matching, artifact counts) is absorbed or null on full-fold reads; only bank pooling moved OOF (0.95194 → 0.95252). Untested own avenues: replica e160+ with lr 2e-4 (memorisation regime the generator likely sat in), replica *initialising* our short-format train LM, GReaT-format full leg (stopped after 1h, resumable).
| 11-04 | replica e120 (from e80, +40 ep at lr 2e-4), full fold 1 | solo 0.82020 | loglik -123.7 (still improving) but pool_v4 gain **-0.00003** -> FAIL. Deeper memorisation hurts solo and adds nothing; orig-only generator replica closed at e10-e120, lr 5e-5 to 2e-4 |

## Day 5 (2026-11-05)

- 08:58 PC had rebooted overnight (GReaT ep10 leg died at step 6.5k of fold 1; resumable). LB top 0.95263
  (Hawksartelier); top public notebooks are now rank means of public *files* (goodpjw Part 10 0.95239 =
  0.5 Part 9 + 0.5 amanatar Sovereign file, no CV). goodpjw's honest nested CV is 0.95199; ours 0.95267.
- Harvest: kernels → `ext/ko5/` (new OOF: mikhailnaumov XGB2_12, kava1 xgb_v5, georgymamarin v-new);
  datasets → `ext/ds5/`, csv/npz → `ext/ds5_conv/` via `src/conv_ds5.py` (sadamtorres TabPFN-3.5 +
  GBDT pairs, queyrusi digit CNN, wguesdon trees, malyshev cross-fitted, digit-token joint) +
  goodpjw series-artifacts new version (test-bag members).

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-05 | pool_v6 = v5 sources + ko5 | 0.952532 | ≈ v5 (noise) |
| 11-05 | **pool_v7** = v6 + ds5 series-artifacts + ds5_conv (170 members after dedup) | **0.952670** | +0.00014 over v6 (3.5 sd); without sadam_* 0.952571 (+0.0001 from their v6 gb/glm pairs), without digitcnn 0.952644, without sadam TabPFN-3.5 0.952670 (0); ours +0.000056; **submitted** |
| 11-05 | exact / leave-one-column-out duplicate rows across train+test | 0 groups | the generator never repeats a row; dead |
| 11-05 | LM per-class logliks (l1, l0 separately; + log P(x); + order spread) vs LLR on top of pool_v7, ep10 f1 40k K=16 | −0.00009 / −0.00012 / −0.00002 | LLR is the right summary; dead |

**PR-1 (11-05 09:36, before the number): Deotte's full Storyteller recipe — replica-initialised GReaT LM.**
Arm: distilgpt2 initialised from our orig-only GReaT replica `great_e40.pt` (40 ep on the 20k original
rows), then label-first GReaT-format fine-tune on fold-1 train, lr 5e-5, 8 epochs, bs 32, 25% negatives,
K=3 val orders (`lm_leg.py --great_format --orig_epochs 40 --lr 5e-5 --epochs 8 --tag _great_r40_ep8`).
Paired control on the other Kaggle T4: identical without the init (`--tag _great_lr5_ep8`).
Instrument: fold-1 solo AUC + `lm_pool_eval.py`-style fold-1 gain over pool_v7's logit.
**Bar:** replica-init f1 solo ≥ 0.9494 (masa's public GReaT LM f1 0.94888 + 0.0005) **and** fold-1 gain over
pool_v7 ≥ +0.0001 → full 5-fold leg. Init effect = init − control (paired). Prior ~15%: our Day-2
short-format orig pre-fit (10 ep, then 2 ep/fold at lr 2e-4) was noise, but that was neither GReaT format nor a
low fine-tune lr, which are the parts of the recipe that would preserve the replica.
PR-1 arm 3 (added 09:45, before any number, same bar): local RTX 2060, replica-e40 init at our tuned lr 2e-4,
10 epochs (`--great_format --orig_epochs 40 --epochs 10 --bs 32 --tag _great_r40_ep10`), fold 1.
| 11-05 | pool_v8 = v7 + goodpjw full 70-member `s6e11-oof-bank` (`ext/ds5b`; 218 members after dedup) | 0.952631 | below v7; without the bank 0.952666; C .003/.005 → .952578/.952622 → the extra bank members are noise; keep v7 |
| 11-05 | stacker variants on the cached v7 matrix (`src/pool_lab.py`, base reproduces 0.952666) | — | nest10 .952661, C .02 .952660, standardized cols .952397, random-subspace bag .95251, LM cols ×1.5/2/3 .952662/.952634/.952634, + masa² .952682 (+0.000016, noise), + top-5² .952583 → all null; plain LR C .01 stays |
| 11-05 | harvest ko6 (6 kernels since 09:00) | — | only georgymamarin re-run (same members as ko5); nothing new |
| 11-05 | naive-Bayes per-value leg (`src/nb_leg.py`, fold-safe smoothed value rates summed as log-odds), raw / + digit keys | 0.92505 / 0.90448 | pool_v7 gain (`src/pool_add.py`, cached matrix) −0.000007 / +0.000014 → absorbed; dead |
| 11-05 | within-fold rank-logit member transform (cached v7) | — | −0.00002 → null; global rank-logit stays |
| 11-05 | class-weighted stacker (balanced C .01 / C .001) | — | −0.00028 / −0.00005 → null |
PR-1 arms 4-5 (added 11:55, before any number, same bar): Kaggle `s6e11-lm-replica-ctl2` — arm 4 = no-init
GReaT lr 2e-4 ep10 (`_great_ep10`, the paired control for local arm 3); arm 5 = replica-e40 init, lr 5e-5,
14 epochs (`_great_r40_ep14`, does the low-lr arm just need more steps?).
| 11-05 | leave-one-source-out ablation (cached v7, 1 seed; noise ±0.00003) | — | only ds5_conv clearly carries signal (drop −0.000125); every other source/member within ±0.00006 → no pruning |
| 11-05 | orig-only GLM (the label formula; `src/orig_glm.py`, no train labels) lin / spline | train 0.93435 / 0.93205 | pool gain −0.000003 / −0.000012 → absorbed |
| 11-05 | kNN leg (k 100, rank-gauss nums + one-hot, ref = all pos + 120k neg reweighted; `src/knn_leg.py`) | 0.91899 | pool gain +0.000007 → absorbed |
| 11-05 | LM consensus column (mean of 5 LM rank-logits) replacing / added to the LM columns | — | −0.000023 / +0.000002 → null |
| 11-05 | ridge (squared-loss) stacker, alpha 1e2-1e6 | — | −0.0047…−0.0019 → LR loss stays |
| 11-05 | harvest ko7 (areebbb pool, megayak re-run, FT-transformer, DBSCAN, flaml) + alexchilton bank update (`ext/ds7`, `src/pool_add_dir.py`) | — | no new OOF members after dedup |
| 11-05 | **PR-1 arm 3** (replica-e40 init, GReaT, lr 2e-4, ep10, local) fold 1 | f1 **0.94823** | short ep10 .94846, masa .94888 → **FAIL** (bar 0.9494); pool_v7 fold-1 gain −0.000098, over base+masa −0.000016. Replica init at our lr does not help (lower train loss = memorises, does not generalise) |
| 11-05 | temperature-scaled LM scoring (ep10 f1 model, 40k, K=4; `src/lm_temp.py`), T .7–1.4 | solo .94541→.94593 (rises with T) | pool_v7 subset gain −0.00005 → −0.0001 (falls with T); pool_noLM same → dead: flatter LLR helps solo only |

**PR-2 (11-05 13:45, before the number): fixed-column-order Storyteller.** Arm: short-format label-first
distilgpt2 trained and scored in ONE column order (CSV order, `--fixed_order`, K=1), 5 epochs, fold 1, 30k
val subset. Reference: random-order ep5 f1 on the same 30k rows. Bar: solo ≥ reference + 0.0005 → full-fold
ep10 run. Prior ~15%: one factorisation concentrates capacity, but GReaT itself trained on random orders, so
the sampled data carry no preferred order and K-order averaging is lost.
| 11-05 | NB raw + NB dig + kNN + orig-GLM together in the pool | — | −0.000002 → null |
| 11-05 | **PR-2 fixed-column-order LM** (short, ep5, K=1, f1 30k) | 0.94358 | random-order ep5 K3 same rows 0.94351 → +0.00007 < bar +0.0005 → **FAIL**; corr .981 with ep5; one order with K=1 ≈ random orders with K=3 (cheaper scoring, same signal) |

**PR-3 (11-05 14:35, before the number): digit-spelled Storyteller.** Numbers written one character per token
(`--spaced_digits`: "1 3 7 1 . 3 7"), short format, label-first distilgpt2, ep5, fold 1 full val, K=3. Rationale:
a different view of the same density (the BPE LMs share number tokens with the generator; this one must learn
digit structure) → possibly decorrelated from masa/ours (rho ~.98). Bar: f1 solo ≥ 0.9468 (ep5 .94733 − 0.0005)
**and** pool_v7 fold-1 gain ≥ +0.0001 → full leg. Prior ~10% (less faithful to the generator's tokenisation).
| 11-05 | **PR-1 arms 1-2** (Kaggle; GReaT lr 5e-5 ep8, replica-e40 init vs none) fold 1 | f1 **0.94812 / 0.94802** | init effect +0.0001 (noise); both < bar 0.9494; pool_v7 gain −0.000065 / −0.000019 → **PR-1 FAIL**. Deotte's replica-init recipe does not transfer: the masa-style LM is at this kind's ceiling (~.9485-.9489 on f1) |
- 11-05 14:40: **Kaggle weekly GPU quota (30 h) exhausted** — local RTX 2060 only until the weekly reset.
| 11-05 | harvest ko8 (queyrusi two-stacks, longnguyen xgb) | — | submissions only, no OOF |
| 11-05 | **PR-3 digit-spelled LM** (short, ep5, f1 full) | f1 **0.94290** | < bar 0.9468 → **FAIL**; pool_v7 f1 gain +0.000020 (ep5 −0.000029), over base+masa +0.000015; **rho(masa) 0.956** vs ≈0.983 for every BPE LM → the most decorrelated LM so far |

**PR-4 (11-05 15:40, before the number): digit-spelled LM, 10 epochs** (`--spaced_digits --epochs 10`, f1 full, K=3).
Rationale: 3× more tokens per row → ep5 likely undertrained (final train loss above BPE ep5). Same bar as PR-3:
f1 solo ≥ 0.9468 **and** pool_v7 fold-1 gain ≥ +0.0001 → full 5-fold leg (local, ~2.5 h/fold → spans into tomorrow). Prior ~15%.
| 11-05 16:30 | harvest (spline-tree-rankstack update; others submissions-only) | — | no new members; LB top 0.95269 |
| 11-05 | **PR-4 digit-spelled LM ep10** f1 | f1 **0.94495** | ep5 .94290 → +0.002 from length, rho(masa) .964; pool_v7 f1 gain +0.000034 (over base+masa +0.000046) → **FAIL** (bar .9468 / +0.0001). Gain grows ~+0.000015 per doubling of epochs: ~3× short of the bar; digit-spelling avenue closed at ep5-ep10 |

### 11-05 hand-off
- **Best OOF + submitted: pool_v7 0.952670** (`src/pool_stack.py --ext <v5 dirs> ext/ko5 ext/ds5/s6e11-fraud-series-artifacts
  ext/ds5_conv`, C .01, dedup .9999, min_auc .93). Cached matrix for 5-min stacker/member tests: `results/pool_lab.npz`
  (`src/pool_lab.py cache|run`, `src/pool_add.py <member>`, `src/pool_add_dir.py <dir>`).
- Dead today (all pre-registered or cheap-screened): replica-init LM at lr 5e-5/2e-4 (PR-1), fixed-order LM (PR-2),
  digit-spelled LM ep5/ep10 (PR-3/4; most decorrelated LM, rho .96, but gain +0.00003), LM temperature, split
  logliks, NB, kNN, orig-only GLM, 12 stacker variants, goodpjw full bank (overfits), duplicate rows.
- Kaggle GPU quota exhausted until the weekly reset. Still in flight there: `s6e11-lm-replica-ctl2` (arms 4-5,
  decision-irrelevant now; log them when it completes: `kaggle kernels output alessandrosanvito/s6e11-lm-replica-ctl2`).
- What still moves the pool: new public same-fold OOF banks (ds5_conv batch +0.0001). Re-harvest every few hours. Next full rebuild should add `ext/ko10 ext/ko11_conv` (each ~+0.00003, below noise alone).
- **RESUME:** PR-6 (digit-spelled LM lr 5e-4 ep20, f1) runs overnight → read it first (command in PR-6). Untested own avenues: a digit-spelled
  GReaT-format LM; stacking at test time on refits only.

**PR-5 (11-05 17:45, before the number): digit-spelled LM, lr 5e-4, ep10** (≈ ep20 at lr 2e-4; on BPE LMs lr 5e-4
halved the epochs needed). f1 full val, K=3, no test. Same bar as PR-3/4: solo ≥ 0.9468 and pool_v7 f1 gain ≥ +0.0001.
Honest expectation: gain ~+0.00005 (extrapolating +0.000015 per doubling) → likely FAIL. Prior ~10%.
| 11-05 | **PR-1 arms 4-5** (Kaggle ctl2): arm 4 no-init GReaT lr 2e-4 ep10; arm 5 replica-e40 init lr 5e-5 ep14 | f1 0.94768 / **0.94903** | arm 5 = best fold-1 LM so far (> masa .94888) but < bar .9494 and pool_v7 gain −0.000082 (rho masa .987) → FAIL. Init effect at lr 2e-4: arm 3 − arm 4 = **+0.00055** (at lr 5e-5 ep8: +0.0001). Replica init is a real solo gain but the pool absorbs it (same kind as masa) |
| 11-05 19:00 | harvest ko10: sameershaikh digit-sequence CNN (3 members, solo .945, maxcorr .968) | — | pool delta −0.000031 → null; rest submissions-only |
| 11-05 | **PR-5 digit-spelled LM lr 5e-4 ep10** f1 | f1 **0.94610** | pool_v7 f1 gain **+0.000060** (ep10 lr 2e-4 +0.000034, ep5 +0.000020), rho(masa) .967 → FAIL (bar .9468 / +0.0001), but the only own lever whose pool gain rises monotonically with training |

**PR-6 (11-05 19:40, before the number): digit-spelled LM lr 5e-4 ep20** (`--spaced_digits --lr 5e-4 --epochs 20
--tag _spaced_ep20_lr5`, f1 full, K=3, no test; local, ~3.5 h → finishes ~23:30, resumable). Same bar: solo ≥ 0.9468
and pool_v7 f1 gain ≥ +0.0001 → full 5-fold leg. Expectation from the trend: gain ~+0.00009 → coin flip at best. Prior ~25%.
Read with: `python src/lm_pool_eval.py pool_v7 _lm_distilgpt2_spaced_ep20_lr5 _lm_distilgpt2_spaced_ep10_lr5`.
| 11-05 21:00 | harvest ko11: denpugovkin joint-gains GLM/4-leaf (5) + logreg v2 (parquet → `ext/ko11_conv`) | — | pool delta +0.000031 (< 1 sd) → not submitted; include in the next full pool rebuild |
| 11-05 21:15 | pool_v9 = v7 sources + ko10 + ko11_conv (178 members) | 0.952662 | ≈ v7 (−0.000008) → not submitted; pool_v7 stays the best |
| 11-05 | **PR-6 digit-spelled LM lr 5e-4 ep20** f1 | f1 **0.94627** | pool_v7 f1 gain **+0.000001** (ep10 lr5 +0.000060), over base+masa +0.000017, rho(masa) .957 → **FAIL**; the rising-gain trend broke at ep20 (more training = more masa-like solo, less unique). Digit-spelled avenue closed |

## Day 6 (2026-11-06)
- 08:42 PC rebooted again at 23:25 (PR-6 had finished at 23:12). Session 13:20, deadline for today's window 22:00.
| 11-06 13:50 | harvest ko12 (21 kernels since 21:00: longnguyen xgb_v6 ×6, dhananjayk part 4 blends, masa OOF stack, mikhailnaumov, kava1, yekenot RealMLP csv → `ext/ko12_conv`) | — | pool delta −0.000036 (ko12) / −0.000011 (ko12_conv) → null |
| 11-06 | harvest datasets: masa TabPack MLP bank (0.94402, same folds; `ext/ds12_conv`) | — | pool delta −0.000002 → null. dosomethingbetter GPT-2 replica/story weights not used (training data unknown → leak risk) |
| 11-06 | label autocorrelation along id (lags 1-100), 10k-block fraud-rate sd vs binomial | ≈0 / .00121 vs .00128 | no generation-order signal; dead |

**PR-7 (11-06 14:00, before the number): EBM on raw + digit keys** (`ebm_leg.py --digits`: per float column
first/second decimal, #decimals, int length, leading and last int digit as written in the CSV = 36 label-free
categorical keys; EBM handles them natively, so no target encoding and no fold leak). Rationale: the two own
findings that moved anything were the GA2M structure (EBM best own tree-free leg 0.94846) and the
generator's digit fingerprint (public digit-TE trees 0.9515); they have never been combined in one GA2M.
Fold-1 pilot, inter 30, bags 4, jobs 4. **Bar:** pool_v7 fold-1 gain ≥ +0.0001 (`lm_pool_eval.py pool_v7
_ebm_i30_dig`) → full 5-fold leg. Honest expectation: solo ~0.950-0.951 (between EBM and digit trees),
gain ≈ 0 because the pool holds GLM-boost digit models (same additive kind) → likely FAIL. Prior ~15%.

**PR-8 (11-06 14:05, before the number): discriminatively fine-tuned Storyteller** (`src/lm_disc.py`). Start from the
ep10 fold-1 generative LM; train its Bayes score s(x) = log P(x|1) − log P(x|0) + prior directly with BCE on y
(+ 1.0 × generative NLL as regulariser, lr 2e-5, bs 32, 1:3 pos:neg, 3 passes over positives), on rows the
generative model never saw: fold-1 val, cross-fitted in two stratified halves (fit A → score B and back).
Paired reference: the untouched ep10 model on the same rows/orders (K=3). Rationale: every LM so far is a pure
density fit and the pool absorbs them (same kind as masa); a conditional-likelihood objective shapes the LLR
where the classes overlap, which a density fit does not target. **Bar:** pool_v7 fold-1 gain ≥ +0.0001
(`lm_pool_eval.py pool_v7 _lm_distilgpt2_disc`) **and** disc solo ≥ gen solo → full leg (cross-fitted per fold).
Honest expectation: ~1.2k positives per half is little for 82M params; solo ±0.001, gain ≈ 0. Prior ~15%.
| 11-06 | **PR-7 EBM raw + 36 digit keys** f1 (10 min, 4 jobs) | f1 **0.94724** | raw EBM same seed-class .94706 → digits +0.0002 solo; pool_v7 f1 gain **−0.000014** (over base+masa +0.000009) → **FAIL**. Low-card digit keys barely help a GA2M; the public digit signal lives in high-card per-value TE (int part, token tries), already in the bank |
| 11-06 | **PR-8 discriminative fine-tune of the ep10 LM** (BCE on LLR + gen NLL, cross-fitted fold-1 val halves) | f1 **0.94714** | paired gen 0.94846 → −0.0013 (both halves lower: .94949/.95087, .94497/.94610); pool_v7 gain −0.000023 → **FAIL**. 1.2k positives per half overfit 82M params; the conditional objective does not add a new kind |
| 11-06 | public-member test-file audit (cached v7): corr(member, consensus) test − OOF; OOF-vs-test logit KS | — | no member drops on test (min −0.0011; median +0.002 = fold-averaged test preds are smoother) → no misaligned/broken test files. megayak per-value TE bank (9) has test on another scale (KS .89) — harmless under rank-logit |

**PR-9 (11-06 15:05, before the fold-1 number): factorization machine over value tokens** (`src/fm_leg.py`; full values,
int part, first decimal, int+d1, leading 2 digits, #decimals, last 2 decimals; rare <20 → per-field rare). Tuned on
**fold 2** only (solo: high-card tokens overfit; best .9425 dropping full/ipd1/last2 tokens, .9385 with AdamW wd 10 on
all tokens). Read on fold 1, 2 seeds, both configs. Bar: pool_v7 f1 gain ≥ +0.0001. Expectation: ≈ LR-level solo → absorbed. Prior 5%.
| 11-06 | **PR-9 FM over value tokens** f1 (2 seeds) | drop-highcard **0.94305** / wd10 0.93916 | pool_v7 gain −0.000052 / −0.000030 → **FAIL**. Per-value signal needs TE-style shrinkage; an FM's L2/decay overfits high-card tokens, and without them it is an LR |
| 11-06 15:15 | harvest ko13 (queyrusi part 8 = submissions; nikitachubai LGBM .94886 → `ext/ko13_conv`; lourens GLM-CatBoost = holdout split, not our folds) | — | pool delta +0.000010 → null |
| 11-06 | **pool_v7 residual scan** (honest OOF, 1-param recalibrated): 493 marginal cells (raw bins + digit keys), 61.5k pairwise cells, exact-value + int-part cells of every high-card column, var = Σq(1−q) | max\|z\| 1.82 / 3.96 (null ≈ 3.0 / 4.7); per-value χ²/df ≤ 1 | **no marginal, pairwise or per-value residual structure** — the pool is calibrated everywhere a cell-level lever could live. Same instrument on our ep10 LM alone: z up to +8.7 (customer_age, flags) → it does detect structure |
| 11-06 | per-member leak audit: χ²/df of OOF residuals in (fold, exact value) cells of the 6 high-card floats | most public members 0.99-1.03, ours LM 1.01 / EBM 1.04; under-dispersed only the 10-fold / other-fold members (goodpjw gb10_* .93-.97, sadam v6 pairs .93) → cross-fold neighbours, per-row honest; pool_v7 0.912 inherits it | **no leaky member to prune**. (Aggregating cells over all folds is biased low for any honest cross-fitted TE — the per-fold cell is the right test) |

**PR-10 (11-06 15:15, before the number): discriminative fine-tune on fold-train rows** (`lm_disc.py --fit_on train`):
same objective as PR-8 (BCE on the LLR + 1.0 × gen NLL, lr 2e-5, bs 32, 1:3) but fit on all 9.3k fold-1 train positives
+ the ~75% of fold-1 train negatives the generative ep10 fit never saw, 1 pass over positives; score fold-1 val, K=3.
PR-8 failed on data size (1.2k positives per half); this has 8× the positives at the cost of positives the generative
model already fit (their LLR is inflated → small gradients there). Paired reference: ep10 f1 0.94846. Bar: solo ≥
0.94846 and pool_v7 f1 gain ≥ +0.0001. Prior ~10%.
| 11-06 | **PR-10 disc fine-tune on fold-train rows** (9.3k pos + unseen negs, 1 pass) f1 | f1 **0.94793** | paired ep10 0.94846 → −0.0005; pool_v7 gain −0.000027 → **FAIL**. Discriminative-LM avenue closed (val-halves and train-rows both lose) |
| 11-06 | non-negative LR stacker (L-BFGS-B, cached v7) | 0.952067 | free LR 0.952675 (78 negative coefs, Σneg −2.85 vs Σpos 4.23) → negative weights carry −0.0006 of real signal; free-LR combination spread test sd 2.49 vs OOF 2.50 → no test-side blow-up; keep free LR |

**PR-11 (11-06 15:55, before the number): K=16 column orders on the full fold 1** for the ep10 LM (`lm_rescore.py
_lm_distilgpt2_ep10 --k 16 --sub 139992`, logsumexp). The 11-04 K=16 read used 40k rows (gain resolution ~±0.0001,
i.e. could not see the size of effect that matters now); LMs carry the pool's largest weights (masa .36, ep20 .19,
ep10 .13). Instrument: `lm_pool_eval.py pool_v7` on the K=16 vs the stored K=3 f1 scores (paired, same model).
Bar: K16 − K3 pool_v7 f1 gain ≥ +0.0001 → rescore ep10/ep20 val+test at K=16 for all folds (~8 h GPU, overnight). Prior ~15%.
| 11-07 | **PR-11 K=16 vs K=3 column orders, ep10 LM, full fold 1** (same model, paired) | K3 0.94829 / K16 **0.94855** | pool_v7 f1 gain K16 −0.000035 vs K3 −0.000039 → **+0.000004 ≪ +0.0001 → FAIL**. More orders sharpen the solo, not the pool |

## Day 7 (2026-11-07)
- PC rebooted again overnight (shutdown 23:24). Session 09:50 → 22:00.

**Screen S7-1 (11-07 09:55, before the number): segment-interaction LR stacker** on cached pool_v7 (top-20 |w| members ×
segment indicators + segment intercepts; segments: any ".0" float, #".0" floats, channel, auth_method, card_type).
Nest seed 7 only, paired vs base. Bar: ≥ +0.0001 → confirm on 3 seeds. Expectation: pool already calibrated along the
artifact axis and categoricals (residual scan 11-06) → ≈ 0. Prior 5%.

**PR-12 (11-07 10:10, before the number): drop column-name tokens from the LM Bayes score** (`src/lm_rescore_names.py
_lm_distilgpt2_ep10 --fold 1 --k 3`). Our LLR sums every body token incl. " name" tokens, but column order is our own
shuffle → log P(name | label, prefix) has zero true label signal; its label-conditional part is fit noise in the LLR.
megayak's public LM masks names out ("column tokens are inputs, never targets"). Same model/rows/orders, paired.
**Bar:** nonames − names pool_v7 f1 gain ≥ +0.0001 (`lm_pool_eval.py pool_v7`) → rescore val+test for ep5/ep10/ep20
all folds (~1.5 h GPU each). Expectation: name-term LLR sd small vs LLR spread → solo +0.0001..0.0005, pool gain mostly
absorbed but the masa-decorrelation could help. Prior 20%.
| 11-07 | **S7-1 segment-interaction LR stacker** (cached v7, nest seed 7, base 0.952646) | dot0 −0.000021, dot0n −0.000041, channel −0.000042, auth_method −0.000150 | **FAIL** — segment-specific member weights only add variance; the pool's combination is uniform across segments |
| 11-07 | **PR-12 drop column-name tokens from the LM score** (ep10 f1, same orders, K=3) | names 0.94842 / nonames **0.94840** | name-term LLR sd 0.042 (tiny vs LLR spread); pool_v7 gain −0.000036 vs −0.000035 → **+0.000001 → FAIL**. Note: re-drawing the K=3 orders alone moves the pool gain by 3e-5 (stored −0.000003) = the LM order-noise floor |

**PR-13 (11-07 10:25, before the number): pseudo-likelihood Bayes score** (`src/lm_pll.py _lm_distilgpt2_ep10 --fold 1`,
full fold 1, ~2 h GPU). Same saved ep10 model; for each column c put c last (others random) → log P(x_c | x_-c, y);
PLL-LLR = Σ_c [lp(c|1) − lp(c|0)]. A different estimator of the class contrast than the prefix-chain LLR every public LM
uses (it weights each column by its full-context conditional, not its position in a chain) → plausibly different errors.
By-product: chain LLR at K=24. **Bar:** PLL pool_v7 f1 gain ≥ +0.0001 (`lm_pool_eval.py pool_v7 _lm_distilgpt2_ep10_pll`)
→ score ep5 all folds + test. Expectation: solo below the chain LLR (PLL double-counts dependence), rho(masa) ~.95;
gain likely ≤ +0.00005. Prior 10%.
| 11-07 10:15 | harvest ko14 (25 kernels since 11-06 14:00) + ds14 (goodpjw artifacts 11-07, megayak target-first LM, alexchilton frozen banks, amanatar rank3, nina 03) | — | 4 new members after dedup (megayak glm_boost_c/raw/cnt, kava1 catboost_v7) → pool_v7 delta **−0.000011** → null. megayak LM bank has only folds 0-1 (NaN elsewhere) → not poolable yet |
| 11-07 | goodpjw wide stack (200 members, nested OOF 0.95250) as one pool_v7 member | — | maxcorr .991, delta **−0.000037** → null (it is our pool's twin) |

**PR-14 (11-07 10:50, before the number): class-specialist fine-tune** (`src/lm_specialist.py _lm_distilgpt2_ep10 --fold 1
--epochs 2 --lr 2e-5`). Copy A of the ep10 f1 model fine-tuned on the 9.3k fold-train positives only (prefix 1); copy B
on an equal number of fold-train negatives the generative fit never saw (prefix 0); LLR = log P_A(x|1) − log P_B(x|0).
Rationale: in the joint model log P(x|1) is fit from 1.7% of the tokens and shares all capacity with the negatives; a
short specialist pass sharpens the minority conditional without a discriminative objective (PR-8/10 lost to overfit of
BCE). Also reported: positive specialist only (A vs joint for y=0). Paired ref = joint model, same rows/orders, K=3.
**Bar:** pool_v7 f1 gain of spec (or specpos) − specref ≥ +0.0001 → full leg (5 folds; ep5 models are all local).
Expectation: positives already fit 10 epochs in the joint model → A overfits them further; solo ±0.0003. Prior 10%.
| 11-07 | **test-side audit of pool_v7 vs the public wide stack** (1 − Spearman; OOF vs test, against common members gb_bag5 / masa GPT-2 / our LM ep10 / ebm_dig) | ours oof→test .0127→.0085, .0284→.0270, .0285→.0237, .0228→.0130; wide .0097→.0146, .0246→.0316, .0309→.0325, .0215→.0191 | our stack moves *closer* to members on test (fold-averaged members are smoother) = consistent; goodpjw's wide stack moves *away* on test → the extra ours-vs-public distance on test (.0099 vs .0033 OOF) is their test-side noise (10-fold OOF vs refit test mismatch), not ours. pool_v7's test side is sound; public LB ranks above us need not transfer |
| 11-07 | member OOF-vs-test structure scan (R² of each member on all others, fit on OOF, applied to test; full-data C .01 fit) | LMs: unexplained var OOF/test .035/.016 (ep20), .021/.016 (masa), .014/.005 (ep10) | ⚠️ `results/pool_lab_w_free.npy` stores the **intercept LAST** (not first) — reading it with `[1:]` shifts names by one (a "realmlp_s42 w −6.8" scare was the intercept; true w −0.001). Real finding: LM members are 2-3× noisier on OOF (1 model, K=3) than on test (5 models × K=3) → the stacker is attenuation-biased *against* LMs on test (conservative, not harmful). Dropping realmlp_s42: −0.000011 OOF, test distance .00065 → irrelevant |
| 11-07 | megayak target-first GPT-2 LM bank (compact format, names masked; folds 1-2 only), single-logit gain over pool_v7 | f1 .94851 / f2 .95068 | gain −0.000131 / −0.000034, rho(masa) .984/.982 → null (masa's kind); no need to wait for its other folds |

**PR-15 (11-07 11:35, before the number): negative-quarter LM ensemble** (Kaggle `s6e11-lm-negq`, 2× T4; weekly GPU
quota has reset). Every own LM (ep5/ep10/ep20) was fit on the *same* 25% negative subsample (rng 1000+k, first draw), so
they share their training negatives. New `lm_leg.py --neg_quarter q` fits the disjoint quarter q (q=0 = original). Fold 1,
ep10 recipe (lr 2e-4, bs 64, K=3, same val orders as ep10), quarters 1 and 2. Read: avg(q0,q1,q2) LLR vs q0 alone,
`lm_pool_eval.py pool_v7`. Rationale: 3× the distinct negatives at equal per-model compute — data, not just a seed
(all-negatives single model at equal compute failed 11-04; this is 3× compute). **Bar:** avg pool_v7 f1 gain − q0 gain
≥ +0.0001 → q1-q3 for all folds (~8 kernel-runs). Expectation: solo +0.0005 (ensemble), pool mostly absorbed (masa and
the 3 own LMs already average). Prior 15%.
| 11-07 12:15 | harvest 15: alexchilton oof-predictions update (ds15), denpugovkin less-freedom GLM (parquet → `ext/ko15_conv`, same folds), hirofumitsuda, malyshev fusion, giannhs | — | ds15/ko15: 0 new after dedup; ko15_conv 3 new → delta **−0.000007** → null |
| 11-07 | **PR-13 pseudo-likelihood LM score** (ep10 f1, each column last, full fold) | PLL solo **0.93067**, K=24 chain 0.94852 | PLL rho(masa) **.912** (most decorrelated LM-derived score yet) but pool_v7 gain −0.000025; 24 per-column PLL terms as stacker features −0.00025/−0.00028 (C .01/1) → **FAIL**. Full-context conditionals double-count dependence (solo −0.018); decorrelation without strength again |

### 11-07 status (12:35)
- Best OOF + submitted is still **pool_v7 0.952670**; its test side is sound (audit above). LB top 0.95322 (sihaoli666).
- Kaggle weekly GPU quota has reset (probe ran on T4 11-07). In flight: `s6e11-lm-negq` (q1, q2 fold 1) and
  `s6e11-lm-negq-b` (q3 fold 1, q1 fold 2 + test) for PR-15 → download with
  `kaggle kernels output alessandrosanvito/s6e11-lm-negq -p kaggle_out/negq` into `results/_lm_distilgpt2_ep10_q*/`.
- Local GPU: PR-14 class specialists (`logs/lm_spec_f1.log`).
| 11-07 | epoch-trend read at pool level, f1 (standard format): ep5_lr5 / ep10_lr5 (≈ep20) / ep20 | solo .94793 / .94738 / .94750 | pool_v7 gain +0.000005 / −0.000014 / −0.000061 → past ~ep10-equivalent the single LM overfits; **no ep40 run**. The single-model LM ceiling is reached; only data-diverse LM ensembles (PR-15) remain on this axis |
| 11-07 | **PR-14 class-specialist fine-tune** (ep10 f1; A on 9.3k pos, B on 9.3k unseen negs, 2 ep, lr 2e-5) | ref 0.94836 / spec **0.94545** / pos-only 0.94736 | pool_v7 gain −0.000035 / −0.000034 (ref ≈ −0.00003) → **FAIL**. Single-class fine-tuning drifts each conditional away from the shared calibration the LLR needs (solo −0.003); decorrelated (rho .968) but weaker |
| 11-07 | **PR-15 negative-quarter LM ensemble** f1 (Kaggle; q1, q2 ep10 on disjoint negative quarters; q0 = stored ep10) | q1 .94869 / q2 .94851; avg(q0,q1) .94892; **avg(q0,q1,q2) .94902** (+0.00056 solo) | inter-quarter corr .9917 (= seed-level); pool_v7 gain avg012 −0.000030 vs q0 −0.000003 → **FAIL**. More distinct negatives sharpen the solo LM exactly like a seed bag, and the pool already holds that (masa + ep5/10/20). q3 (Kaggle negq-b) not needed; speculative local q1-f3 run stopped |

**Screen S7-2 (11-07 13:55, before the number): clustered-member stacker** on cached pool_v7: hierarchical clustering of
members on |corr| (average linkage), each cluster collapsed to its mean rank-logit, LR C .01 on cluster means
(thresholds .995/.99/.98), plus "means + all members" at C .01. Paired vs base, 3 nest seeds. Bar ≥ +0.00008 (2 sd).
Expectation: the free LR already shrinks within near-duplicate groups via L2 → ≈ 0 or slightly negative. Prior 5%.
| 11-07 | **S7-2 clustered-member stacker** (avg-linkage on \|corr\|; cluster-mean columns) | .995: 84 cl. −0.000158 / +members −0.000001; .99: 51 cl. −0.000224 / +0.000005; .98: 28 cl. −0.000523 / −0.000007 | **FAIL**. Collapsing even corr>.995 near-duplicates costs 0.00016 → the differences *between* near-duplicate members carry real signal (cf. 11-06 negative weights worth 0.0006); the free LR over all members is the right combiner |
| 11-07 14:10 | harvest 16: lucifer19 golden-signal v5/v8 (3-fold own split, digit/token TE; npz → `ext/ko16_conv`), denpugovkin train-verify GLM/tree (parquet), sometimessubodh xgb, dhananjayk wide stack 18 | — | 10 new (gs_v8 maxcorr .980 = most decorrelated) → pool_v7 delta **−0.000027** → null. Note: never `np.load(..., allow_pickle=True)` on downloaded files (code execution) — the classifier blocks it; plain arrays load without it |

**Screen S7-3 (11-07 14:15, before the number): pairwise-ranking stacker** (AUC surrogate) on cached pool_v7: within each
inner train split, every positive paired with 20 random negatives; LR without intercept on ±(x_pos − x_neg), C ∈
{.01, .1, 1}; scored on the inner val split; same nest seeds (7, 13, 99) as lr_stack. Also 0.5·rank(pair) + 0.5·rank(LR).
Bar ≥ +0.00008 (2 sd) over base. Expectation: log-loss LR and pairwise LR agree for near-logistic signal → ≈ 0. Prior 7%.
| 11-07 | **S7-3 pairwise-ranking stacker** (20 negs/pos, LR on ±differences, no intercept; nest seeds 7/13/99) | C .01 / .1 / 1: −0.000222 / −0.000479 / −0.000535; 50/50 rank mix with LR −0.000004 | **FAIL**. The log-loss LR beats the AUC surrogate (a pairwise fit sees only 20 negatives per positive, and the full negative ranking matters) |

### 11-07 hand-off (14:30)
- **Best OOF + submitted: pool_v7 0.952670** (unchanged). Test side audited sound vs the public wide stack.
- Dead today (all pre-registered, fold-1 or cached-matrix reads): K=16 orders (PR-11), name-token masking (PR-12),
  pseudo-likelihood score (PR-13), class specialists (PR-14), negative-quarter LM ensemble (PR-15), epoch trend past
  ep10-equiv, segment-interaction stacker (S7-1), clustered-member stacker (S7-2), pairwise-ranking stacker (S7-3),
  megayak LM bank, goodpjw wide stack as a member, harvests 14/15/16 (all ≤ 0).
- **The LM axis is exhausted at pool level**: every LM change that raises solo AUC (orders, data, epochs, ensembles)
  is absorbed (masa + ep5/10/20 already average it); every LM change that decorrelates (PLL, specialists, digits)
  loses too much solo. Only a *new kind* or new public banks can move pool_v7.
- Kaggle GPU quota reset Sat 11-07 (~4.5 h used of ~30). Kernel `s6e11-lm-negq-b` (q3 f1, q1 f2) still running; not
  needed after PR-15 FAIL.
- Tools added: `lm_leg.py --neg_quarter`, `lm_rescore_names.py`, `lm_pll.py`, `lm_specialist.py`.
| 11-07 14:35 | **pool_v10** = v9 sources + ko12, ko12_conv, ko13_conv, ds12_conv, ko14, ds14 (goodpjw, frozen), ko15_conv, ko16, ko16_conv (drop wide_stack) | 0.952615 | −0.000055 vs v7 at C .01 → not submitted; C re-tune on the v10 matrix next |
| 11-07 | pool_v10 C re-tune (197 members, cached `results/pool_lab_v10.npz`) | C .02 .952613 / .01 .952615 / .005 .952609 / .003 .952585 / .002 .952529 | flat-to-worse at every C → the 26 extra members are noise for the free LR; **pool_v7 stays best** (171 members). Harvest 17 (gaurav TabPFN chunks = no output, lucifer v13 = stacks/blends only, susilnayak = submission) → nothing to add. `results/pool_lab.npz` restored to the v7 matrix |

**Screen S7-4 (11-07 14:45, before the number): importance pruning of pool_v7 members** (cached v7). Importance =
|coef| × sd(column) from the full C .01 fit; drop the bottom k ∈ {20, 40, 60, 90} and re-run lr_stack (3 nest seeds).
Selection uses the full-data fit → mildly optimistic; bar ≥ +0.00008 (2 sd) to adopt. v10 showed extra members cost
−0.00005, so pruning could recover ~that. Prior 10%.
| 11-07 | **S7-4 importance pruning of pool_v7** (drop bottom-k by \|coef\|·sd of the full fit) | k 20/40/60/90: +0.000032 / +0.000023 / +0.000017 / +0.000029 | below the +0.00008 bar (and optimistic: importances from the full fit) → not adopted. But the bottom 90 of 171 members carry **nothing**: an 81-member pool equals v7 → a simpler finals hedge candidate (`pool_v7_p81`, build with nested importance selection if used) |

**Screen S7-5 (11-07 15:00, before the number): class-conditional latent-class mixture** (`src/lcm_leg.py`, fold 1):
P(x|y) = Σ_k π_k Π_j θ_kj(x_j), numerics in 32 fold-train quantile bins, cats/bools as codes; EM (K0 64 on a 25%
negative subsample, K1 16 on all positives, Dirichlet α 1, 40 iters); score log P(x|1) − log P(x|0). A generative
density that is neither an LM nor independent (NB 0.925 was absorbed). Bar: pool_v7 f1 gain ≥ +0.0001 → 5 folds.
Expectation: solo 0.93-0.94, rho(pool) low, absorbed like NB/PLL. Prior 5%.
| 11-07 | **S7-5 latent-class mixture leg** (`src/lcm_leg.py`, f1) K0 64/K1 16, 32 bins; K0 200/K1 32, 64 bins | solo **0.93387** / 0.93171 | rho(masa) .89; pool_v7 gain −0.000040 / +0.000001 → **FAIL**. Non-LM generative density beats NB (.925) but is too weak; more components overfit the 9.3k positives |
| 11-07 15:20 | harvest 19: lucifer19 shadow-vote v14 | — | no OOF files (submission only) → nothing to add |
| 11-07 17:05 | harvest 20: areebbb GLM-boost + residual MLP pool | — | submission only, no OOF → nothing to add |
| 11-07 20:15 | harvest 23: megayak target-first GPT-2 LM bank **complete** (5 folds; `ext/ds23/megayak-lm`), megayak shallow-tree stack re-run | LM solo **0.95043** (> masa .95003) | maxcorr .982; pool_v7 delta **+0.000039** (~1 sd) → not submitted; include in the next rebuild (candidate pool_v11 = v7 + megayak LM) |
| 11-07 21:00 | **pool_v11** = v7 sources + megayak complete LM bank (`ext/ds23/megayak-lm`) | **0.952705** | +0.000035 over v7 (< 1 sd, bar +0.0001) → **not submitted**; best OOF file on disk, finals candidate. Submit only when the next public bank batch stacks on top of it to ≥ +0.0001 over v7 |
| 11-07 21:10 | **pool_v11 submitted** (user asked; best OOF 0.952705, needed on the board to be selectable as a final) | | |

## Day 8 (2026-11-08)
- Session 15:30 → 22:00. LB top 0.95328 (강북구집돌이); goodpjw Part 12 (wide stack + megayak LM, nested 0.95252) LB 0.95298.
| 11-08 15:50 | harvest 25a: sometimessubodh 38-feature bank (12 members, `ext/ds25_conv/subodh`), mathsonchain 17 single models (same folds), goodpjw artifacts 11-07 23:52 (realmlp_*), + megayak LM | — | 28 new; pool_v7 delta +0.000031 vs megayak alone +0.000039 → **new banks −0.000008 = null** |
| 11-08 16:00 | harvest 25b: alexchilton shallow-tree-stack re-run (38 oof_hash members), yekenot RealMLP v2 (0.95012, csv → `ext/ko25_conv`) + megayak LM | — | 3 new; delta +0.000049 → **+0.00001 over megayak = null**. Others (goodpjw part 12, sameershaikh wide stack, malyshev, georgymamarin, denpugovkin, gaurav AutoGluon) = stacks/submissions only |

**PR-16 (11-08 16:10, before the number): top-k truncation evidence** (`src/lm_topk.py _lm_distilgpt2_ep10 --fold 1 --k 3`).
GReaT samples at top-k 50: a value token outside the generator's top 50 under label y's context cannot be emitted with y →
hard evidence the LLR sees only smoothly. Same ep10 f1 model, same orders as PR-12 (rng 777); per value token the rank of the
true token in the restricted vocab under each label; per row: #tokens with rank ≥ 10/20/50/100 per label (mean/min over
orders), max log-rank, #tokens out of top-50 under y but inside under 1−y. Read: LR on [pool_v7 logit, ep10 LLR, features]
vs [pool_v7 logit, ep10 LLR], inner 5-fold CV on fold 1. **Bar:** ≥ +0.0001 → score all folds + test (only the f1 model
is saved locally → would need retraining = Kaggle). Expectation: our short-format LM ranks are a poor proxy of the
generator's GReaT-format ranks under its unknown order; most of the information is already in the LLR. Prior 7%.
| 11-08 16:20 | harvest 25c: kava1 catboost v9 (6, .946), mikhailnaumov XGB2_12 (.9501) + megayak LM | — | delta +0.000028 (< megayak alone) → null |
| 11-08 | **PR-16 top-k truncation evidence** (ep10 f1, K=3, rng 777; 32 rank features) | LLR solo 0.94842; best single feature nout20_d auc .69 | base+LLR+feats: all C .1 **−0.000121**; best group nout20 **+0.000036**, excl +0.000035 → **FAIL** (bar +0.0001). Under the *true* label our LM still puts 0.58 tokens/row outside its top 50 (rank ≥ 100: 0.03) → its ranks are a loose proxy of the generator's top-k set under an unknown order; what signal there is already lives in the LLR |

**Screen S8-1 (11-08 16:50, before the number): today's banks on the importance-pruned pool** (`src/pool_pruned_add.py`).
v7 pruned to its top 81 members (S7-4: = v7), then megayak LM + harvest 25a/b/c members added unselected. Rationale: the
free LR over 171 members may drown small new signals that a leaner pool can use. Bar: pruned81 + new ≥ v7 + 0.0001
(selection on the full fit is mildly optimistic → only the new-member delta is clean). Prior 10%.
| 11-08 | **S8-1 today's banks on pruned-81 v7** (36 new after dedup) | pruned81 0.952695; + all new **0.952729** | per source: yekenot RealMLP v2 +0.000030, megayak LM +0.000026, subodh +0.000010, catboost −0.000008, mathsonchain −0.000021. All new = +0.000063 vs v7, i.e. +0.00003 over pool_v11 (and pruning selection is optimistic) → **FAIL** (bar +0.0001). Leaner pool does not unlock the weak banks. ⚠️ `lr_stack` drove load to 16 on 12 cores (BLAS threads) — run pool jobs with `OMP_NUM_THREADS=4` |

**PR-17 (11-08 17:35, before the number): non-GPT-2 pretrained Storyteller — SmolLM2-135M** (`lm_leg.py` now loads any
causal LM: `AutoModelForCausalLM`, `model.base_model`, fp32 weights). Same label-first short format / 25% negatives /
restricted softmax as ep5/ep10; SmolLM2's tokenizer writes every digit as its own token (`123.45` → 1,2,3,.,4,5) and it
was pretrained on a different corpus (FineWeb-Edu/code/math). Every LM in the pool is GPT-2-family (ours, masa, megayak);
PR-3/4's digit-spelled GPT-2 was the most decorrelated LM (rho masa .96) but GPT-2 never saw digit-split numbers in
pretraining. Kaggle `s6e11-lm-smollm` 2×T4, fold 1, `--val_sub 50000 --k_val 3 --no_test`: arm A ep3, arm B ep5 (lr 2e-4,
bs 64). Read: pool_v7 single-logit gain on the 50k subset, paired against ep10 (and ep5) GPT-2 on the same rows.
**Bar:** SmolLM gain − ep10 gain ≥ +0.0001 → full leg on Kaggle (~4 h/fold/GPU). Expectation: solo ≈ ep5 GPT-2, rho(masa)
~.96-.97, gain absorbed like PR-3/4. Prior 5%.

**Screen S8-2 (11-08 16:55, before the number): local re-stack of the contested region** (`src/pool_local.py`, cached v7).
Per nest split: global LR C .01; region = top q of the global train score (q .8/.9/.95); a local LR (C .01/.1) fit on region
rows only re-scores val region rows (mapped linearly onto the global scale). Rationale: the global log-loss fit is dominated by
easy negatives while AUC is decided where positives meet hard negatives. Bar ≥ +0.00008 (2 sd). Expectation: LR over near-
logistic members is already locally right; s6e7 cascade was null. Prior 5%.
| 11-08 | **S8-2 local re-stack of the top-q region** (cached v7, nest 7/13/99) | global 0.952666 | q .8/.9/.95 × C .01: −0.000017 / −0.000002 / +0.000007; C .1: −0.000096 / −0.000037 / −0.000009 → **FAIL**. The global LR is already right where positives meet hard negatives; a local fit only adds variance |
- PR-17 arms C/D — **DEFERRED** (Kaggle GPU slots both busy: v1 of s6e11-lm-smollm kept running after the v2 push; run only if A/B show promise) (same bar, same read; kernel `s6e11-lm-smollm-b` built): C = SmolLM2-135M lr 5e-4 ep3 (GPT-2: higher lr ≈ half
  the epochs); D = SmolLM2-360M lr 2e-4 ep1.5 bs 32 (capacity on the new family; GPT-2 124M = distilgpt2, so prior lower).
| 11-08 | **S8-3 test-side attenuation check** (`src/attenuation_check.py`; free LR fit on folds 2-5, fold 1 scored with the ep10 column = q0 vs avg(q0,q1,q2) quantile-mapped) | (a) 0.952099 / (b) 0.952106 | denoised LM input +0.000007; scaling the ep10 coef λ 1.2/1.5/2: −0.000013/−0.000033/−0.000070 → the 11-07 "attenuation against LMs on test" worry is immaterial; no correction |

**Screen S8-4 (11-08 17:25, before the number): GBDT residual over the pool members** (`src/pool_gbdt_resid.py`): LightGBM
(4/8 leaves, lr .02, min_child 2000/4000, ff .5, L2 10) boosted from the nested LR stack OOF (init_score) on the 171 v7
rank-logit members; outer 5-fold, inner 10% early stopping. Never tried at pool scale (day-3 GBDT meta over 9 legs lost
0.0005). Bar ≥ +0.00008. Expectation: no member × member structure left (residual scans clean) → early stop at ~0 trees. Prior 5%.
| 11-08 | **S8-4 GBDT residual over pool members** (init = nested LR OOF) | LR 0.952666 | 4 leaves −0.000032 (best_iter 1,1,75,255,9), 8 leaves −0.000067 → **FAIL**. Early stopping finds nothing in 2/5 folds; no member × member interaction left for a tree to use |
- PR-17 instrument calibration (before the number): on the 50k fold-1 subset (861 pos), the paired gain difference
  ep5 − ep10 GPT-2 = +0.000107 with row-bootstrap sd **0.000085** → the +0.0001 bar is ~1.2 sd; a pass is only a trigger
  to rescore the full fold 1 with the kept model (sd ~half), not a verdict.
| 11-08 17:20 | harvest 26: giannhs boosting-bagging-neurals (new), georgymamarin re-run, tigerpiotta EDA | — | no OOF files → nothing to add |
| 11-08 | S8-1 honest re-read: importance pruning selected **inside** each nest split (`src/pool_pruned_nested.py`) | v7 0.952666 / v7+36 new (free) 0.952702 / **nested-pruned81+new 0.952654** | pruning is worth −0.000012 when nested → S8-1's 0.952729 (and S7-4's "pruned81 = v7") carried ~+0.00008 selection optimism. Do not use full-fit importances for pool selection; no pruned finals candidate |
| 11-08 | C re-tune on v7 + 36 new members (incl. megayak) | — | C .005/.01/.02: +0.000020/+0.000036/+0.000031 vs v7 → flat; C .01 stays; nothing beyond pool_v11 |
| 11-08 19:05 | harvest 27: giannhs re-run (log only), sameershaikh wide stack re-run (members.csv: same 14 sources we already pool — goodpjw, alexchilton, sameershaikh, malyshev, lazyyeou, sadam, megayak, nilanjan, masa, queyrusi, yekenot, wguesdon), gaurav AutoGluon, baselines | — | nothing new to add |
| 11-08 | **PR-17 SmolLM2-135M Storyteller** (Kaggle 2×T4, fold 1, 50k-row subset, K=3; ~29 min/1k steps bs 64, 20 min scoring) | ep3 solo **0.94551** / ep5 **0.94633** (GPT-2 ep5 .94677, ep10 .94856 on the same rows) | pool_v7 gain −0.000025 / −0.000059 vs ep10 −0.000094, ep5 +0.000033 → paired vs ep10 +0.00007/+0.000035 (sd .000085), vs ep5 negative → **FAIL**. rho(masa) .958 = as decorrelated as the digit-spelled GPT-2 (PR-3/4) and, like it, weaker solo; gain falls with epochs. A digit-native pretrained tokenizer is the same story as spelling digits in GPT-2: a new view, not a stronger one. Arms C/D (lr 5e-4, 360M) not worth running. Models in `results/_lm_SmolLM2-135M_ep{3,5}/f1_model.pt` |

### 11-08 hand-off (20:00)
- **Best OOF + submitted: pool_v11 0.952705** (v7 + megayak LM). Every public nested CV is still below it (goodpjw Part 12
  .95252, sameershaikh wide ~.9525). LB top 0.95328; public LB noise ~7e-4 (goodpjw), so LB ranks above us need not transfer.
- Closed today (pre-registered): harvests 25-27 (36 new members, all ≤ +0.00001 over megayak), PR-16 top-k truncation
  evidence, PR-17 SmolLM2 (non-GPT-2 digit-native LM), S8-1 pruned pool (nested: −0.000012), S8-2 local re-stack,
  S8-3 test-side LM attenuation (immaterial), S8-4 GBDT residual over members, C re-tune.
- What this measured: the combiner is right (LR, C .01, all members; no local, interaction, pruning or nonlinear gain),
  and every new LM kind tried (tokenizer, pretraining corpus, scoring rule) trades solo strength for decorrelation at
  ~zero net. Untested classes that remain: a *stronger-solo* LM (megayak's format beat ours by +0.0005 solo — the only LM
  change that ever moved the pool was solo strength, ep2→ep5→ep10), and new public banks.
- Next session: re-harvest first (`kaggle kernels list --sort-by dateRun`, `kaggle datasets list --search s6e11 --sort-by
  updated`); measure with `src/pool_add_dir.py ext/ds23 <new dirs>` (subtract megayak's +0.000039). Kaggle GPU quota:
  ~7 h used this week (resets Sat). Run pool CPU jobs with OMP_NUM_THREADS=4.
| 11-08 21:15 | harvest 28: areebbb wide-stack re-run, lucifer v22, parth catboost, kospintr/kiza baselines; no new datasets | — | stacks/submissions/known weak members only → nothing to add. LB top 0.95328 (unchanged since 12:03) |

## Day 9 (2026-11-11)
- Session 09:00 → 22:00 (3 days since the last one). LB top 0.95369 (tine.sh agent); the public board is now led by
  **"ladders"** (goodpjw LM Ladder 0.95346, areebbb Three Ladders 0.95347): a 0.95340 file + LB-probed steps
  `A + 0.6(A − F) + 0.12(megaLM − F) + 0.06(LM7 − F)` in logit ranks. Their own authors report every step **loses on
  CV** (−1.5e-5 … −14.7e-5 on the wide-stack OOF) and tell readers to pair them with a CV-backed pick. Best public
  CV-backed numbers: amanatar "sovereign" 0.952572, goodpjw wide stack 0.952565 — still below pool_v11 0.952705.
| 11-11 09:30 | harvest 29 (datasets): megayak LM (OOF unchanged, only test re-scored), goodpjw `s6e11-oof-bank` (143 files) + sk20220623/alexchilton `s6e11-oof-predictions` + amanatar vault (`ext/ds29`, 51 new after dedup) | — | v7 + megayak + all new **−0.000020** (megayak alone +0.000039) → null. amanatar vault alone on v7+megayak (seed 7): s24 BPE-pair chain (solo .9434, maxcorr .944) −0.000010, trigen LM fusion −0.000069, 4 "control variates" −0.000087, 7 "vetoes" −0.000100 → null. goodpjw seed-7 compact LM: fold 0 only (f0 .947545) → not poolable yet; re-check |
- Deotte's s6e10 table (`cdeotte/kaggle-playground-solutions`): GPT-2 small target-first 0.962425, medium 0.962370,
  large 0.962360 → **a larger GPT-2 is not a lever**; label-anywhere small 24 ep 0.962452 was his best single, and the
  *same* model read target-last scored 0.960961 (our pure label-last Detective: 0.943 vs 0.948 label-first).
- Running downloaded scripts locally is blocked by the permission classifier (megayak's `lm_member.py`), so the
  compact format is re-implemented in our own `src/clm_leg.py` (one learned token per column, softmax over the 1,372
  used value tokens + END + 2 labels, rows assembled on the GPU; 67 tokens/row vs ~130 in `lm_leg.py`). Local RTX 2060:
  distilgpt2 bs 64 451 rows/s fit, gpt2 bs 32 204 rows/s (bs 64 OOMs at 6 GB).

**PR-18 (11-11 09:35, before the number): label-anywhere compact LM** (`src/clm_leg.py --label any`). The label is a
25th field at a uniform random position during the fit, so one model learns p(y | any prefix) *and* p(fields | y, prefix).
Three reads of the same fold-1 model, K=4 draws each: `first` (Storyteller LLR), `last` (Detective, a discriminative
read that has seen the generative objective — Deotte's s6e10 version lost only 0.0015 to the Storyteller, ours pure
label-last lost 0.005), `any` (label at a random position: prefix posterior + suffix LLR). distilgpt2, 25% negatives,
ep10, lr 2e-4, bs 64 = the recipe of our `_lm_distilgpt2_ep10` (f1 0.9484, K=3) → paired reference.
Read: `lm_pool_eval.py pool_v7 _clm_any_ep10_{first,last,any} _lm_distilgpt2_ep10`, plus first+last jointly.
**Bar:** best single read, or first+last together, pool_v7 f1 gain − ep10 gain ≥ +0.0001 → 5-fold leg (Kaggle T4s,
all negatives). Honest expectation: `first` ≈ 0.948, `last` ≈ 0.945-0.947, all three rho(masa) ≥ .96 and absorbed
(17 LM variants so far were); the only new thing is the Detective read from a jointly trained model. Prior 7%.
| 11-11 09:55 | harvest 29 (kernel outputs since 11-08 21:00, 24 kernels → `ext/ko29`, csv/npz → `ext/ko29_conv`, `ext/ko29_tab`): akmalshahbaz joint GLM + shallow GBDT (7), kava1 xgb_v12, yekenot RealMLP v3 (.95002), staesthetics LGB TE + Benford digits (.9429, maxcorr .960), herczeg TF MLP (.9423, maxcorr .952), zhenrui XGB, slimypunk TabPFN starter (5 folds, .94929, maxcorr .982) | — | 14 new + megayak: **+0.000007** (megayak alone +0.000039) → null. Rest (denpugovkin 5-reviews, georgymamarin, samanyu ladders/pair-risk, queyrusi two stacks, malyshev fusion, kazuhiroyogo) = submissions/stacks only |
| 11-11 | **ladder move on our own OOF** (fold 0, logit ranks): pool_v11 + w·(LM − pool_v11) for megayak LM / goodpjw seed-7 LM (not in our pool) / their average | w .03/.06/.12/.2: mega −0.000039/−0.000081/−0.000176/−0.000328; lm7 −0.000027/−0.000062/−0.000148/−0.000298; avg2 −0.000032/−0.000068/−0.000152/−0.000287 | stepping toward an LM **loses at every step size**, also toward an LM the pool has never seen and toward the 2-seed average (the test-side denoising argument) → the public LM steps repair files that hold too little LM (Strike 10 carries ≈ −0.001 LM); pool_v11 already has the LM weight right. No ladder on our files |
| 11-11 | pool_v11 ⊕ amanatar sovereign (best public CV-backed stack, .952572; corr OOF .9965 / test .9946) rank-logit blend | w .1/.2/.3/.5: +0.000012/+0.000019/+0.000019/+0.000004 | null. Note for finals: the public 0.95340 ladder file correlates only **.9934** with pool_v11 on test — the public top and our stack are materially different files, so a ⊕ blend is a real hedge option for final #2 (no OOF exists for it; decide at finals time, deadline 11-30) |
- PR-18 arms on Kaggle 2×T4 (registered 09:52, before any number; kernel `s6e11-clm-pr18`, fold 1, same bar and read):
  **B** = compact `--label first` distilgpt2 ep10 (format control: compact vs our short-name `_lm_distilgpt2_ep10`, and
  first-read of a label-first fit vs of a label-anywhere fit); **C** = `--label any` with gpt2 small (bs 32, ep10).
  Instrument note: a fold-1 pilot can only resolve ≥ +0.0001. New `src/fold_member_eval.py` scores a column inside the
  full member matrix on the fold's rows (megayak's LM reads −0.00004 there, +0.00004 in the full pool; known-null
  variants q1/q2/r40/PLL read −0.00004…−0.00008) → use it next to `lm_pool_eval.py`; a pass needs ≥ +0.0001 on it.
| 11-11 | **public-LB noise for ladder-sized steps** (300 random 60k-row OOF samples ≈ the public 20%; paired AUC delta of two files) | sovereign + 0.12(megaLM − sov): full-OOF −0.000070, 60k sd **0.000069**; + 0.06: −0.000029, sd 0.000034; + 0.6(sov − newton) extrapolation: +0.000003, sd 0.000082; pool_v11 vs pool_v7: +0.000035, sd 0.000109; pool_v11 vs sovereign: +0.000133, sd **0.000237** (P(ours ahead on a public-size sample) 0.70); one file's AUC sd 0.0031 | every public ladder step (+0.00001…+0.00003 on the LB) is **< 1 sd** of its own public sampling noise → 0.95313 → 0.95347 is what ~10 accepted noise-selected steps look like; it is not evidence of signal. The public LB also cannot rank pool_v11 against the best public CV stack (0.6 sd); on the private 80% the same +0.00013 is ~1.1 sd. Finals stay CV-ranked |
| 11-11 | **learning curve in positives vs negatives** (XGB d3 raw, max_bin 2048, fold 1, best of 300-1500 trees) | pos 100% / 50% / 25% (9.35k / 4.67k / 2.34k) with all negatives: **0.94663 / 0.94489 / 0.94278**; with 25% of the negatives: 0.94607 / 0.94491 / 0.94287 | each halving of the positives costs ~0.002; 4× the negatives buys ≤ 0.0006 (0 below 9k positives) → **the task is positive-limited** (9.35k per fold; the generator itself saw 339). Consequences: negative subsampling in LM fits is nearly free; whatever uses the positives more efficiently (generative or hybrid objectives, label-free structure from the 440k unlabeled val+test rows) is the class with headroom; "more members" of a discriminative kind is not |
