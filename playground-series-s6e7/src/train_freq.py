"""LGBM leg with frequency-encoding features — the last unmeasured FE family.

Motivated by the Day-4-night grid finding: the numerics are coarse quantized
grids (water 400 uniques, heart_rate exactly 1dp, calorie/steps integer). A
value's population frequency is DATASET-level information a tree cannot compute
from the row itself, so unlike every row-wise transform already measured flat
(ordinals, target encoding, interactions, indicators, driver posteriors) this
injects something genuinely new. Payoff mechanism = label-correlated sampling
density, i.e. generator artifacts -- the artifact probes were empty, so expect
small; this run makes FE fully measured rather than mostly measured.

Features: per-numeric value counts over train+test pooled (transductive,
label-free; NaN rows stay NaN) + the 3-driver combo count (64-way, NaN as a
level). Label model = baseline.py verbatim. Run paired with lgbm_r_s42:

  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 \
      ../.venv/bin/python src/train_freq.py

Gate: weighted OOF delta vs lgbm_r_s42 (0.9478) > +0.001 solo or blend-level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from baseline import run
from features import NUM_COLS
from train_common import Dataset, finalize, load_dataset

COMBO = ["stress_level", "physical_activity_level", "sleep_quality"]


def add_frequency_features(ds: Dataset) -> None:
    full = pd.concat([ds.train, ds.test], ignore_index=True)
    n_train = len(ds.train)

    new_tr: dict[str, np.ndarray] = {}
    new_te: dict[str, np.ndarray] = {}
    for col in NUM_COLS:
        if col not in full.columns:
            continue
        counts = full[col].value_counts()  # NaN excluded -> NaN maps to NaN
        freq = full[col].map(counts).astype("float32").to_numpy()
        new_tr[f"freq_{col}"] = freq[:n_train]
        new_te[f"freq_{col}"] = freq[n_train:]

    key = full[COMBO].astype("string").fillna("NA").agg("|".join, axis=1)
    combo_freq = key.map(key.value_counts()).astype("float32").to_numpy()
    new_tr["freq_combo"] = combo_freq[:n_train]
    new_te["freq_combo"] = combo_freq[n_train:]

    ds.train = pd.concat(
        [ds.train, pd.DataFrame(new_tr, index=ds.train.index)], axis=1
    )
    ds.test = pd.concat([ds.test, pd.DataFrame(new_te, index=ds.test.index)], axis=1)
    ds.feature_cols = list(ds.train.columns)


def main() -> None:
    ds = load_dataset()
    add_frequency_features(ds)
    n_new = len([c for c in ds.train.columns if c.startswith("freq_")])
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape} (+{n_new} freq features)")

    oof_proba, test_proba, fold_scores = run(ds)
    finalize("lgbm_freq", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
