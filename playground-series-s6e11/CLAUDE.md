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

## Current best

See results/runs.jsonl / experiments log below.

## Experiments log

| Date | Run | OOF AUC | Notes |
|---|---|---|---|
| 11-01 | xgb_s42 (d6) | 0.94675 | baseline, submitted |
| 11-01 | xgb_orig_s42 | 0.94676 | + orig rows: flat |
| 11-01 | xgb_d4 | 0.94726 | |
| 11-01 | xgb_d3 | 0.94758 | shallower is better |
