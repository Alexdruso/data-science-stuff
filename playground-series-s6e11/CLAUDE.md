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
