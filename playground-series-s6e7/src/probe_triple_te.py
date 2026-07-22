"""Exact-value RULE-TRIPLE TE canary (Day-18 evening pitch P1).

The archaeologist kill-shot pinned the label channel to exactly 3 dimensions:
(stress_level x physical_activity_level x EXACT sleep_duration). Nothing in
the feature set encodes that JOINT at exact resolution: te13 carries
per-column marginals, Day-9 combo2 used sleep BUCKETS, Day-10 rule-pair TE
tested PAIRS at exact value (flat). The full triple ~6,300 cells x ~100 train
rows/cell — EB smoothing estimable — hands the trees P(y|triple) directly
instead of asking 33-leaf-regularized splits to compose it from marginals.

Design: train_hgbc_te3 recipe verbatim + ONE extra string column in the TE
view (the exact triple key; any-NaN component -> its own level via 'nan'
strings) -> 3 extra TE cols. Paired 5-fold vs the recorded twin
hgbc_te3_u_s42; instruments argmax + robust-wt refit. Screen bar +0.0007;
escalation = realmlp consumer + 12-cell seat. Honest prior LOW (pair-flat,
kNN-flat lean against; this is the last theory-motivated unmeasured cell).

Run: python src/probe_triple_te.py -> results/probe_triple_te.txt
Resumable per fold via results/cache/ckpt_tripte/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    N_CLASSES,
    N_FOLDS,
    RESULTS_DIR,
    load_dataset,
    robust_decision_weights,
)
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle_utils import weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
TWIN = "hgbc_te3_u_s42"
TRIPLE = ["sleep_duration", "stress_level", "physical_activity_level"]
CKPT = CACHE_DIR / "ckpt_tripte"


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    # the ONE change vs the twin: exact triple key as a 14th TE column
    te_tr_view = te_tr_view.assign(
        _triple=ds.train[TRIPLE].astype(str).agg("|".join, axis=1)
    )
    te_te_view = te_te_view.assign(
        _triple=ds.test[TRIPLE].astype(str).agg("|".join, axis=1)
    )
    n_cells = te_tr_view["_triple"].nunique()
    med = int(te_tr_view["_triple"].value_counts().median())
    say(f"triple key: {n_cells} distinct cells in train (median {med} rows/cell)")
    cols = [*raw_cols, "_triple"]
    te_names = [f"te_{c}_{k}" for c in cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
            axis=1,
        )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    CKPT.mkdir(parents=True, exist_ok=True)
    oof = np.zeros((len(ds.y), N_CLASSES))
    for f, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        fck = CKPT / f"f{f}.npy"
        if fck.exists():
            oof[val_idx] = np.load(fck)
            say(f"  fold {f}: loaded checkpoint")
            continue
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
        model.fit(assemble(X.iloc[tr_idx], te_tr), ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES))
        p_val = model.predict_proba(
            assemble(X.iloc[val_idx], enc.transform(te_tr_view.iloc[val_idx]))
        )
        np.save(fck, p_val)
        oof[val_idx] = p_val
        say(f"  fold {f}: n_iter={model.n_iter_}  "
            f"bacc {balanced_accuracy_score(ds.y[val_idx], p_val.argmax(1)):.4f}")
        del model

    twin = np.load(RESULTS_DIR / f"oof_{TWIN}.npy")
    for name, p in (("twin", twin), ("triple", oof)):
        w = robust_decision_weights(ds.y, p)
        say(f"{name:7s} argmax {balanced_accuracy_score(ds.y, p.argmax(1)):.5f}  "
            f"wt {balanced_accuracy_score(ds.y, weighted_predict(p, w)):.5f}")
    say("screen bar +0.0007 paired; escalation = realmlp consumer + 12-cell seat")
    (RESULTS_DIR / "probe_triple_te.txt").write_text("\n".join(lines) + "\n")
    print(f"report -> {RESULTS_DIR / 'probe_triple_te.txt'}", flush=True)


if __name__ == "__main__":
    main()
