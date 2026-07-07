"""Quantify the cost of the train->test missingness-mechanism shift (plan #0a).

Fits the baseline LGBM per fold (seed 42, same recipe as the champion's LGBM leg)
and scores each validation fold on three surfaces:

  plain     -- val as-is (train mechanism: masks only on trigger rows)
  testmech  -- iid Bernoulli(r_col) masking of OBSERVED values in the 4 shifted
               columns at the TEST marginal rates (gender/trigger-independent,
               like test). Off-trigger conditionals then match test exactly; the
               train-mechanism NaNs already present cannot be unmasked, so
               on-trigger rows keep an unavoidable excess.
  control   -- volume-matched: the SAME number of new NaNs per column, but placed
               only on observed ON-TRIGGER rows (train-mechanism placement).

bacc(plain) - bacc(testmech) estimates the damage; the testmech-vs-control gap
isolates mechanism (placement) from volume (more NaNs). Decision weights are fit
on the plain OOF only (that is what deployment does) and applied to all surfaces.

Run: ../.venv/bin/python src/diag_mask_shift.py | tee results/diag_mask_shift.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from train_common import (
    DATA_DIR,
    N_CLASSES,
    Dataset,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle_utils import weighted_predict

# column -> trigger condition (train mechanism: masks occur ONLY where trigger holds)
TRIGGERS: dict[str, tuple[str, str]] = {
    "water_intake": ("gender", "female"),
    "physical_activity_level": ("smoking_alcohol", "occasional"),
    "bmi": ("sleep_quality", "good"),
    "diet_type": ("gender", "other"),
}


def test_rates() -> dict[str, float]:
    te = pd.read_csv(DATA_DIR / "test.csv")
    return {c: float(te[c].isna().mean()) for c in TRIGGERS}


def remask(
    X_val: pd.DataFrame,
    rates: dict[str, float],
    rng: np.random.Generator,
    mode: str,
) -> pd.DataFrame:
    """Return a copy of X_val with new NaNs per `mode` ('testmech'|'control')."""
    out = X_val.copy()
    for col, (trig_col, trig_val) in TRIGGERS.items():
        if col not in out.columns:  # e.g. water dropped via env
            continue
        observed = out[col].notna().to_numpy()
        r = rates[col]
        if mode == "testmech":
            hit = observed & (rng.random(len(out)) < r)
        else:  # control: same expected count, on-trigger placement only
            n_new = int(round(observed.sum() * r))
            on_trig = observed & (out[trig_col].to_numpy() == trig_val)
            pool = np.flatnonzero(on_trig)
            take = min(n_new, len(pool))
            hit = np.zeros(len(out), dtype=bool)
            hit[rng.choice(pool, size=take, replace=False)] = True
            if take < n_new:
                print(
                    f"    [control] {col}: pool exhausted "
                    f"({take}/{n_new} new NaNs placed)"
                )
        idx = out.index[hit]
        out.loc[idx, col] = np.nan
    return out


def main() -> None:
    ds: Dataset = load_dataset()
    rates = test_rates()
    print(f"test marginal rates: { {k: round(v, 4) for k, v in rates.items()} }")

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    params = load_params()

    n = len(ds.y)
    oof = {m: np.zeros((n, N_CLASSES)) for m in ["plain", "testmech", "control"]}
    changed = np.zeros(n, dtype=bool)  # rows that received >=1 new NaN (testmech)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        model = LGBMClassifier(**{**params, "random_state": 42})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES))

        X_val = X.iloc[val_idx]
        rng = np.random.default_rng(1000 + fold)
        val_test = remask(X_val, rates, rng, "testmech")
        val_ctrl = remask(X_val, rates, np.random.default_rng(2000 + fold), "control")

        cols = [c for c in TRIGGERS if c in X.columns]
        changed[val_idx] = (
            val_test[cols].isna().to_numpy() & X_val[cols].notna().to_numpy()
        ).any(axis=1)

        for mode, xv in [("plain", X_val), ("testmech", val_test), ("control", val_ctrl)]:
            oof[mode][val_idx] = model.predict_proba(xv)
        b = balanced_accuracy_score(ds.y[val_idx], oof["plain"][val_idx].argmax(1))
        print(f"  fold {fold}: plain argmax bacc {b:.4f}")

    # deployment-style scoring: weights fit on PLAIN oof, applied everywhere
    weights = robust_decision_weights(ds.y, oof["plain"])
    print(f"\ndecision weights (fit on plain OOF): {weights.round(4)}")
    print(f"\n{'surface':<10} {'argmax':>8} {'weighted':>9}")
    scores = {}
    for mode in ["plain", "testmech", "control"]:
        a = balanced_accuracy_score(ds.y, oof[mode].argmax(1))
        w = balanced_accuracy_score(ds.y, weighted_predict(oof[mode], weights))
        scores[mode] = w
        print(f"{mode:<10} {a:>8.4f} {w:>9.4f}")

    print(
        f"\nDAMAGE (plain - testmech, weighted): "
        f"{scores['plain'] - scores['testmech']:+.4f}"
    )
    print(
        f"mechanism-only (control - testmech, weighted): "
        f"{scores['control'] - scores['testmech']:+.4f}"
    )

    # focused: rows that received a new NaN under testmech
    print(f"\nrows with >=1 new NaN (testmech): {changed.sum():,} ({changed.mean():.3%})")
    for mode in ["plain", "testmech"]:
        pred = weighted_predict(oof[mode][changed], weights)
        acc = float((pred == ds.y[changed]).mean())
        bacc = balanced_accuracy_score(ds.y[changed], pred)
        print(f"  {mode:<10} on-changed-rows acc={acc:.4f} bacc={bacc:.4f}")


if __name__ == "__main__":
    main()
