"""Gate the missingness-mechanism repairs on the re-masked instrument (plan #0b).

Same fold loop and val surfaces as diag_mask_shift.py — identical fold splits
(seed 42) and identical val remask RNG streams (1000/2000 + fold), so scores are
exactly paired with the baseline instrument. The difference: TRAINING folds are
repaired before fitting.

  r2a -- in-place uniform re-mask: Bernoulli(r_col) masking of observed values in
         the 4 shifted columns at the TEST marginal rates. Training rows then
         carry uniform NaNs on top of the trigger NaNs -- the same mixture the
         testmech val surface has, so the training mechanism matches the
         (emulated) test mechanism and the NaN<->trigger coupling is broken.
  r2b -- concat augmentation: original training fold + a re-masked copy (2x rows,
         2x fit time; keeps every clean row intact alongside its masked twin).

Baseline (no repair) reference = results/diag_mask_shift.txt:
  plain 0.9487 / testmech 0.9467 / control 0.9477 (weighted).
GATE: repaired testmech weighted bacc >= baseline testmech + 0.001, with the
plain surface not cratered.

Also reported: decision weights fit on the model's own TESTMECH OOF and applied
to the testmech surface -- the "fix the weight-fitting surface" half-repair,
deployable on its own even if retraining is a wash.

Run: ../.venv/bin/python src/diag_repair.py r2a | tee results/diag_repair_r2a.txt
Env: S6E7_REPAIR_MULT (float, default 1.0) scales the re-mask rates.

Day-7 addition — the TESTVOL read (volume-honest, Day-6 adv-AUC 0.6886 follow-up):
train/test marginal NaN rates are ~EQUAL per column, so both the r2a repair and the
testmech val surface roughly double NaN volume in the 4 columns vs true test. But for
val rows COMPLETE in all 4 columns, the testmech re-mask is exactly the true-test
surface: uniform mechanism at exact test volume. "testvol" = testmech restricted to
that subset (~86% of rows; biased toward the non-triggered population, identical
subset across mult runs -> paired). Compare mult runs on it; no stored baseline.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from diag_mask_shift import TRIGGERS, remask, test_rates
from features import NUM_COLS
from train_te_num import encode_fold, with_te
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    Dataset,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle_utils import weighted_predict

# weighted-bacc reference surfaces from results/diag_mask_shift.txt (no repair)
BASELINE = {"plain": 0.9487, "testmech": 0.9467, "control": 0.9477}
GATE = 0.001


def repair_train(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    rates: dict[str, float],
    rng: np.random.Generator,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the repaired training fold for `mode` ('r2a'|'r2a_te'|'r2b')."""
    remasked = remask(X_tr, rates, rng, "testmech")
    if mode in ("r2a", "r2a_te"):
        return remasked, y_tr
    if mode == "r2b":
        X_aug = pd.concat([X_tr, remasked], ignore_index=True)
        return X_aug, np.concatenate([y_tr, y_tr])
    raise SystemExit(f"unknown repair mode {mode!r} (expected r2a|r2a_te|r2b)")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "r2a"
    mult = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))

    ds: Dataset = load_dataset()
    rates = {c: r * mult for c, r in test_rates().items()}
    print(f"repair mode: {mode}  rate mult: {mult}")
    print(f"re-mask rates: { {k: round(v, 4) for k, v in rates.items()} }")

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    params = load_params()

    n = len(ds.y)
    oof = {m: np.zeros((n, N_CLASSES)) for m in ["plain", "testmech", "control"]}
    changed = np.zeros(n, dtype=bool)  # rows that received >=1 new NaN (testmech)
    complete4 = np.zeros(n, dtype=bool)  # rows fully observed in the 4 shifted cols

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        X_tr, y_tr = repair_train(
            X.iloc[tr_idx], ds.y[tr_idx], rates, np.random.default_rng(3000 + fold), mode
        )
        if fold == 1:  # overshoot diagnostic: repaired-train NaN volume vs test
            print("  repaired-train NaN rates vs test marginal:")
            for col in TRIGGERS:
                if col in X.columns:
                    print(f"    {col:<26} {X_tr[col].isna().mean():.4f} "
                          f"vs test {test_rates()[col]:.4f}")
        # identical val surfaces to diag_mask_shift.py (same RNG streams)
        X_val = X.iloc[val_idx]
        val_test = remask(X_val, test_rates(), np.random.default_rng(1000 + fold), "testmech")
        val_ctrl = remask(X_val, test_rates(), np.random.default_rng(2000 + fold), "control")

        X_val_fit = X_val
        if mode == "r2a_te":
            # Exact-value numeric TE (train_te_num protocol: inner cross-fit for the
            # training rows, full-fold maps for val/test) computed on the REPAIRED
            # matrix — exactly what a mult-scaled TE rebuild would deploy. Val
            # surfaces are TE-mapped with the same full-fold maps so the paired
            # TESTVOL/testmech reads stay comparable to the m050/m100 chains.
            num_cols = [c for c in NUM_COLS if c in X.columns]
            # encode_fold only reads numeric columns; the repair remasks bmi (and
            # water, which TE excludes), so hand it a numeric-only frame whose
            # training rows carry the repaired values.
            X_rep = X[num_cols].copy()
            X_rep.iloc[tr_idx] = X_tr[num_cols].to_numpy()
            te_tr, full_maps, prior = encode_fold(X_rep, ds.y, tr_idx, num_cols, 42)
            X_tr = pd.concat([X_tr, te_tr], axis=1)
            X_val_fit = with_te(X_val, full_maps, prior)
            val_test = with_te(val_test, full_maps, prior)
            val_ctrl = with_te(val_ctrl, full_maps, prior)
            X_val = X_val_fit

        model = LGBMClassifier(**{**params, "random_state": 42})
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val_fit, ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES))

        cols = [c for c in TRIGGERS if c in X.columns]
        changed[val_idx] = (
            val_test[cols].isna().to_numpy() & X_val[cols].notna().to_numpy()
        ).any(axis=1)
        complete4[val_idx] = X_val[cols].notna().all(axis=1).to_numpy()

        for m, xv in [("plain", X_val), ("testmech", val_test), ("control", val_ctrl)]:
            oof[m][val_idx] = model.predict_proba(xv)
        b = balanced_accuracy_score(ds.y[val_idx], oof["plain"][val_idx].argmax(1))
        print(f"  fold {fold}: plain argmax bacc {b:.4f} (best_iter {model.best_iteration_})")

    tag = "" if mult == 1.0 else f"_m{int(round(mult * 100)):03d}"
    for m, arr in oof.items():
        np.save(RESULTS_DIR / f"oof_repair_{mode}{tag}_{m}.npy", arr)

    # deployment-style: weights fit on the repaired model's PLAIN oof
    weights = robust_decision_weights(ds.y, oof["plain"])
    print(f"\ndecision weights (fit on plain OOF): {weights.round(4)}")
    print(f"\n{'surface':<10} {'argmax':>8} {'weighted':>9} {'vs-baseline':>12}")
    scores = {}
    for m in ["plain", "testmech", "control"]:
        a = balanced_accuracy_score(ds.y, oof[m].argmax(1))
        w = balanced_accuracy_score(ds.y, weighted_predict(oof[m], weights))
        scores[m] = w
        print(f"{m:<10} {a:>8.4f} {w:>9.4f} {w - BASELINE[m]:>+12.4f}")

    # half-repair: fit the weights on the testmech surface instead
    weights_tm = robust_decision_weights(ds.y, oof["testmech"])
    tm_tm = balanced_accuracy_score(ds.y, weighted_predict(oof["testmech"], weights_tm))
    print(f"\nweights fit on TESTMECH oof -> testmech surface: {tm_tm:.4f} "
          f"(vs plain-fit {scores['testmech']:.4f}; weights {weights_tm.round(4)})")

    # TESTVOL: volume-honest read = testmech restricted to rows complete in the
    # 4 shifted columns (uniform mechanism at EXACT test volume there). Paired
    # across mult runs (same subset, same RNG streams); no stored baseline.
    c4 = complete4
    print(f"\nTESTVOL (complete-in-4 rows: {c4.sum():,} = {c4.mean():.1%}):")
    for wname, wvec in [("plain-fit", weights), ("testmech-fit", weights_tm)]:
        tv = balanced_accuracy_score(
            ds.y[c4], weighted_predict(oof["testmech"][c4], wvec)
        )
        pv = balanced_accuracy_score(
            ds.y[c4], weighted_predict(oof["plain"][c4], wvec)
        )
        print(f"  {wname:<13} testvol={tv:.4f}  (plain-surface same rows: {pv:.4f})")

    delta = max(scores["testmech"], tm_tm) - BASELINE["testmech"]
    verdict = "PASS" if delta >= GATE else "FAIL"
    print(f"\nGATE (testmech vs baseline {BASELINE['testmech']:.4f}, need >=+{GATE}): "
          f"{delta:+.4f} -> {verdict}")
    print(f"plain-surface cost vs baseline: {scores['plain'] - BASELINE['plain']:+.4f}")

    print(f"\nrows with >=1 new NaN (testmech): {changed.sum():,} ({changed.mean():.3%})")
    for m in ["plain", "testmech"]:
        pred = weighted_predict(oof[m][changed], weights)
        acc = float((pred == ds.y[changed]).mean())
        bacc = balanced_accuracy_score(ds.y[changed], pred)
        print(f"  {m:<10} on-changed-rows acc={acc:.4f} bacc={bacc:.4f}")


if __name__ == "__main__":
    main()
