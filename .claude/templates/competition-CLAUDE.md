<!--
Per-competition CLAUDE.md template. Copy into a new `playground-series-<id>/CLAUDE.md`
(the `new-competition` skill does this) and fill every <PLACEHOLDER>. Delete sections that
don't apply yet, but KEEP the row-order invariant section and fill in the real sort keys.
Model it on the worked example in `playground-series-s6e5/CLAUDE.md`.
-->

# <COMPETITION TITLE> (<id>)

**Task**: <binary classification / multiclass / regression> — predict `<TARGET>` (<plain description>)
**Metric**: <AUC-ROC / RMSE / accuracy / …>
**Deadline**: <YYYY-MM-DD>

---

## ⚠️ CRITICAL: Row-order invariant — always load labels and IDs through build_features()

`features.py::build_features()` sorts every dataframe by `<["key1", "key2", ...]>`. All
`results/oof_<model>.npy` and `results/test_<model>.npy` arrays are stored in this sorted order.

Any script that loads `y` or `test_ids` and combines them with the npy arrays **must** go through
`build_features()`, or predictions silently misalign (→ ~0.5 AUC / garbage submission):

```python
# CORRECT
y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()

# WRONG — raw CSV order ≠ npy order
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()
```

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| train | <N> | <C (incl. target)> |
| test  | <N> | <C> |

Target balance / distribution: <e.g. 80% zeros>. Missing values: <none / which columns>.

### Raw features

<list raw feature columns>

---

## Key EDA findings

- <finding 1 — anomalies, leakage risks, distribution shift, strongest predictors>
- <finding 2>

(Adversarial validation AUC, KS stats, train/test category overlap, etc., as relevant.)

---

## Current best

**<Ensemble / model> OOF: <score>** (<date>) ← current best

| Model | OOF <metric> | Script |
|---|---|---|
| LGBM     | <score> | `src/baseline.py` |
| XGBoost  | <score> | `src/train_xgboost.py` |
| CatBoost | <score> | `src/train_catboost.py` |
| MLP      | <score> | `src/train_mlp.py` |

---

## Feature engineering

All features implemented in `src/features.py::build_features()` (+ `compute_group_features` if used).

### What didn't help

| Idea | Δ <metric> | Why |
|---|---|---|
| <idea> | <delta> | <reason> |

---

## Modelling notes

- **CV**: <5-fold stratified, random_state=42>. Scores logged to `results/cv_scores.csv`.
- **GPU**: <which models use GPU and how>.
- **GPU memory rule (all PyTorch scripts)**: `del` model + optimizer + tensors before returning
  from each fold; call `torch.cuda.empty_cache()` after each fold/trial in the outer loop.
- **Categoricals**: <category dtype for LGBM/XGB, cat_features for CatBoost, OHE for nets>.
- **Tuned params**: <Optuna trials, key hyperparameters, which feature set they were tuned on>.

---

## Next steps (remove each item when implemented)

1. <highest-value next experiment>
2. <next>

---

## Experiments log

| Date | Run | Description | OOF <metric> |
|---|---|---|---|
| <YYYY-MM-DD> | <run id> | <what changed> | <score> |
