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

## Current best (2026-11-01 21:25)

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
| 11-01 | ebm_i30_s7 (seed 7) | fold4 0.94793 | (i30 0.94782, i60 0.94784); remaining folds finish after 22:00 |
| 11-01 | **stack_v2** (+ebm_i60, lgbm) | **0.94843** | EBM weights .34/.33; submitted. Stacker C 0.01-100 insensitive |

**Next session:** (1) assemble/stack `ebm_i30_s7`; (2) EBM knobs not yet tried — outer_bags 8+,
`max_interaction_bins`, `smoothing_rounds`, `learning_rate`; give it all cores, one heavy job at
a time; (3) day-1 GReaT fingerprint check (root lesson 14) and plan a cloud-GPU label-first
GPT-2 leg — 29 days remain.

- **GReaT fingerprint (partial, 11-01):** train values copy orig's text formatting exactly
  (e.g. amount: 92% 2-decimal / 8% 1-decimal, i.e. trailing zero dropped, in train *and*
  orig; velocity/merchant_risk 1 decimal everywhere; one 2-decimal merchant_risk in 700k).
  Consistent with a text LM generator. The GPT-2 tokenizer / label-first checks are still
  TODO (`transformers` is not in the venv).
