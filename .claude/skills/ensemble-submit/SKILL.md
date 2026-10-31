---
name: ensemble-submit
description: Rebuild a competition's ensemble from saved OOF/test arrays (Nelder-Mead weight optimization) and produce a Kaggle submission, enforcing the build_features() row-order invariant. Use when asked to ensemble, blend models, build a submission, or submit to Kaggle, e.g. "rebuild the s6e5 ensemble", "blend the models and submit", "make a submission".
---

# Rebuild the ensemble and submit

Use this after training/updating one or more base models to recompute blend weights and produce a
submission. Reference: `playground-series-s6e5/src/ensemble.py`.

## ⚠️ Row-order invariant — read before touching `y` or `test_ids`

Keyed predictions (`results/oof_<model>.parquet`, written by `kaggle.predictions`) carry their
ids: `load_predictions(RESULTS_DIR, m, ids)` re-orders to `ids` and raises on any missing, extra
or duplicate id. Legacy `results/oof_<model>.npy` arrays are positional, stored in
`build_features()` sort order (they load with a warning). Either way, `y` and `test_ids` **must**
come from `build_features()`, or predictions silently misalign → ~0.5 AUC on the leaderboard.
This bug has shipped before. The only correct pattern:

```python
# CORRECT — same order build_features() gives everywhere
train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
y = train[TARGET].to_numpy()
test_df = build_features(pl.read_csv(DATA_DIR / "test.csv"))
test_ids = test_df["id"].to_numpy()

# WRONG — raw CSV order ≠ build_features() order → misaligned submission
y = pl.read_csv(DATA_DIR / "train.csv")[TARGET].to_numpy()   # do NOT do this
```

If the competition uses `compute_group_features` (or similar) after `build_features`, apply it
exactly as the training scripts do so the alignment matches.

## Steps

1. **Reconstruct `y`, `train_ids` and `test_ids`** via `build_features()` as above.
2. **Load OOF/test predictions** for every model in the blend
   (`list_predictions(RESULTS_DIR)` lists what is on disk):
   ```python
   from data_science_stuff.kaggle.predictions import load_folds, load_predictions

   oofs = [load_predictions(RESULTS_DIR, m, train_ids) for m in MODELS]
   tests = [load_predictions(RESULTS_DIR, m, test_ids, split="test") for m in MODELS]
   ```
   Before stacking, check the members share folds:
   `(load_folds(RESULTS_DIR, a, train_ids) == load_folds(RESULTS_DIR, b, train_ids)).all()`.
3. **Optimize weights** with the shared Nelder-Mead search:
   ```python
   from data_science_stuff.kaggle.blending import blend, optimize_blend_weights

   w = optimize_blend_weights(oofs, y, score_fn, normalize="clip")   # or "softmax"
   blended_oof = blend(oofs, w)
   ```
   `score_fn(y, blended) -> float` (higher is better) carries the metric — AUC on 1-D probas,
   `balanced_accuracy_score(y, blended.argmax(1))` for multiclass, negated RMSE for regression.
   Conditional blending (separate weights per data regime, e.g. an `is_2023` mask) is the
   caller's job: pre-slice per mask and optimize each slice, as in s6e5 `ensemble.py`.
   Before pruning/adding bases, print
   `kaggle.blending.diversity_report(oof_arrays, y, class_names, anchor=...)` — error overlap
   vs the anchor predicts blend lift better than probability correlation.
   For multiclass balanced-accuracy tasks, after blending also optimise **per-class threshold
   weights** on the blended OOF and apply the identical weights to test (s6e6: +0.0013):
   `from data_science_stuff.kaggle.decision import optimize_thresholds` → `(weights, score)`.
4. **Report** the blended OOF score and the per-model weights, and register the ensemble with
   its lineage: `save_cv_result(RESULTS_DIR, name, fold_scores, oof, metric_name=...,
   params={"weights": dict(zip(MODELS, w))}, inputs=MODELS, artifacts=[submission_path])`.
5. **Write the submission**:
   `write_submission(SUBMISSIONS_DIR, name, test_ids, TARGET, preds)` from
   `data_science_stuff.kaggle.io` (owns the mkdir, returns the path).
6. **Submit (only when the user asks)**:
   ```bash
   kaggle competitions submit -c <id> -f submissions/<name>.csv -m "<message>"
   ```
   `Bash(kaggle *)` is allow-listed. Confirm the message/filename with the user before submitting.

## When the scalar blend plateaus: stack (s6e6 lesson)

Nelder-Mead assigns **one scalar weight per model**, so a model that is strong on one class and
weak on another gets ~0 weight — no scalar captures "use its GALAXY column, ignore its STAR
column". When the blend stops moving:

1. **Stack with a multinomial LR on the base models' probability LOGITS** — one weight per
   (model, class), fold-honest, multi-seed averaged:
   ```python
   from data_science_stuff.kaggle.stacking import stack_oof

   oof, test = stack_oof(base_oofs, base_tests, y,
                         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
                         seeds=(2024, 7, 13, 42, 99), use_logits=True)
   ```
   Any fit/predict_proba meta works (Ridge-on-one-hot via a small wrapper, a regularized GBDT).
   Worked example: `playground-series-s6e6/src/build_lr_stack.py` (broke a 0.9662 plateau →
   0.9705).
2. **Prune the base set to strong, diverse bases.** Near-duplicate variants add only
   collinearity; on s6e6 six weak ~0.95 bases added +0.0001 = dead weight / overfit surface.
   Ablate before keeping (`diversity_report` shows who fixes whose errors).
3. **If the stack itself is flat, the missing ingredient is a diverse base, not a better meta**
   — a different feature space, loss/model class, or problem decomposition (see the diversity
   section of the `add-model` skill). Adding more same-family models never moved it.
4. **Close out the combiner axis with `kaggle.stacking.caruana_select`** (greedy forward
   selection with replacement, honest outer CV) — if its held-out mean can't beat the stack,
   probability re-weighting is exhausted and the next lever is a new base.

## Optional: post-ensemble blending

Once model training plateaus, `playground-series-s6e5/src/blend_submissions.py` has rank-blending
and selective-consensus-correction utilities. They only help when a support submission is
genuinely diverse (correlation to anchor < ~0.998); if every candidate has corr > 0.999 the signal
is exhausted and micro-blends are noise.

## Verify

The printed blended OOF metric should be ≥ the best single-model OOF. Sanity-check the submission:
correct row count (= n_test), `id` column present, probabilities in `[0, 1]` (for AUC tasks), and
the first few rows match the expected `test_ids` order. If the OOF looks strong but a previous
Kaggle score collapsed to ~0.5, suspect the row-order invariant first.

At competition end, choose which submissions to select with the `select-finals` skill — by CV,
not by public leaderboard rank.
