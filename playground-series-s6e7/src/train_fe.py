"""Feature-engineering probe for PS S6E7 — MEASURE (don't assert) whether FE helps the GBDT.

build_features() is a raw pass-through; the log's "generic FE = noise" was never preserved as a
run. This adds the two principled, non-redundant transforms for THIS problem and gates them:
  1. Ordinal encoding of the 4 ordinal categoricals (stress/sleep_quality/activity/smoking) — the
     natural order the unordered native-categorical handling doesn't get for free.
  2. Fold-safe TARGET ENCODING of the (stress_level, physical_activity_level, sleep_quality) combo
     — this triple IS the label-generating rule (EDA), so a smooth per-class posterior over it is
     the single feature most likely to add signal. Computed on the train fold only (val/test mapped
     from it), smoothed toward the global class prior → no leakage.
Raw columns are kept alongside (trees tolerate redundancy). Gate lgbm_fe with diag_mlp_transfer.py:
must clearly lift adv-weighted / test-like OOF, else FE is measured-flat here (bounded by the
+0.0002 complete∩test-like headroom).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from features import CLASSES
from train_common import N_CLASSES, Dataset, bagged_cv, finalize, load_dataset

ORDINAL = {
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
}
COMBO = ["stress_level", "physical_activity_level", "sleep_quality"]
TE_SMOOTH = 50.0  # pseudo-count toward the global class prior


def add_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, m in ORDINAL.items():
        out[f"{col}_ord"] = df[col].map(m).astype("float32")  # NaN preserved
    return out


def combo_key(df: pd.DataFrame) -> pd.Series:
    return df[COMBO].astype("string").fillna("NA").agg("|".join, axis=1)


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = add_ordinals(ds.train), add_ordinals(ds.test)
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    key_all = combo_key(ds.train)
    key_test = combo_key(ds.test)
    prior = np.bincount(ds.y, minlength=N_CLASSES) / len(ds.y)
    params = load_params()

    def te_maps(tr_idx: np.ndarray) -> dict:
        d = pd.DataFrame({"k": key_all.iloc[tr_idx].to_numpy(), "y": ds.y[tr_idx]})
        enc = {}
        for c in range(N_CLASSES):
            agg = d.assign(t=(d["y"] == c).astype(float)).groupby("k")["t"].agg(["sum", "count"])
            enc[c] = (agg["sum"] + TE_SMOOTH * prior[c]) / (agg["count"] + TE_SMOOTH)
        return enc

    def apply_te(keys: pd.Series, enc: dict) -> np.ndarray:
        return np.stack([keys.map(enc[c]).fillna(prior[c]).to_numpy() for c in range(N_CLASSES)], axis=1)

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001,ANN202
        enc = te_maps(tr_idx)
        te_tr = apply_te(key_all, enc)  # full-train mapped; only tr_idx rows used for fit
        te_val = te_tr[val_idx]
        te_te = apply_te(key_test, enc)
        te_cols = [f"te_{c}" for c in CLASSES]
        Xtr = X.iloc[tr_idx].assign(**dict(zip(te_cols, te_tr[tr_idx].T)))
        Xval = X.iloc[val_idx].assign(**dict(zip(te_cols, te_val.T)))
        Xte = X_test.assign(**dict(zip(te_cols, te_te.T)))
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(Xtr, ds.y[tr_idx], eval_set=[(Xval, ds.y[val_idx])],
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(Xval), model.predict_proba(Xte)

    return bagged_cv(ds, fit_fold, seeds=[42])


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}")
    print(f"FE: +4 ordinal cols, +3 target-encoded cols on {COMBO}")
    oof, test, fold_scores = run(ds)
    finalize("lgbm_fe", ds, oof, test, fold_scores)


if __name__ == "__main__":
    main()
