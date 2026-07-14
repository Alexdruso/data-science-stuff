"""Missingness TTA probe (Day-12 PM) — the last un-run Day-10 backlog item.

Question: does averaging each val row's prediction over K re-masked copies
(test-time augmentation on the missingness channel) beat the single forward
pass? Mechanism: the water-drop lesson — in the near-zero-signal missing-driver
region, tiny NaN-channel perturbations move the at-risk<->minority boundary;
averaging over "this observed value could also have been missing" draws (the 4
MECHANISM_SHIFTED cols at their TEST marginal rates) smooths that tie-breaking.

Consistency requirement (why this is not two lines): the te13 block encodes the
raw value INCLUDING its NaN level ('nan' string = own TE category), so a masked
copy must flip BOTH the raw cell (numeric->NaN, cat->'missing') AND its 3 te13
posterior columns to the fold's nan-level encoding — otherwise the model still
sees the value through the TE channel. The nan-level encoding is recovered from
val rows whose value was ORIGINALLY missing (constant per fold by construction).

Vehicle: the deployed hgbc_te3 leg (Susol config), seed-42 5-fold, u surface,
cached te13 blocks. Paired read: same fits, plain vs TTA-averaged OOF, each arm
getting its own decision weights (identical treatment). Expected flat (trees
estimate P(y|x_obs, pattern) directly under MCAR; marginalization probes lost
2.1pp) — but this exact lever was never measured.

Run: ../.venv/bin/python src/probe_miss_tta.py | tee results/probe_miss_tta.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_common import MECHANISM_SHIFTED, N_CLASSES, N_FOLDS, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from tune_hgbc_te3 import build_fold_designs

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SEED = 42
K = 8  # TTA copies per val row


def nan_level_encoding(X_va, va_orig_nan, te_cols):  # noqa: ANN001, ANN201
    """The fold's nan-level te13 triple for one column, from originally-NaN val rows."""
    rows = np.flatnonzero(va_orig_nan)
    assert len(rows) > 0, "no originally-NaN val rows to recover the encoding from"
    vals = X_va.loc[rows, te_cols].to_numpy()
    ref = vals[0]
    assert np.allclose(vals, ref, atol=1e-5), "nan-level encoding not constant"
    return ref


def main() -> None:
    ds = load_dataset()
    num_cols = [c for c in ds.feature_cols if c not in ds.cat_cols]
    shifted = [c for c in MECHANISM_SHIFTED if c in ds.feature_cols]
    rates = {c: float(ds.test[c].isna().mean()) for c in shifted}
    print(f"TTA cols + test rates: { {c: round(r, 4) for c, r in rates.items()} }  K={K}")

    y = ds.y
    oof_plain = np.zeros((len(y), N_CLASSES))
    oof_tta = np.zeros((len(y), N_CLASSES))
    orig_nan_any = ds.train[shifted].isna().any(axis=1).to_numpy()

    for fold in range(1, N_FOLDS + 1):
        X_tr, X_va, y_tr, val_idx = build_fold_designs(ds, [fold])[fold]
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
        model.fit(X_tr, y_tr)
        assert list(model.classes_) == list(range(N_CLASSES))
        oof_plain[val_idx] = model.predict_proba(X_va)

        # per-column nan-level info, recovered once per fold
        col_info = {}
        for col in shifted:
            te_cols = [f"te13_{col}_{k}" for k in range(N_CLASSES)]
            va_orig_nan = ds.train[col].iloc[val_idx].isna().to_numpy()
            col_info[col] = (
                te_cols,
                va_orig_nan,
                nan_level_encoding(X_va, va_orig_nan, te_cols),
            )

        acc = np.zeros((len(val_idx), N_CLASSES))
        for k in range(K):
            Xk = X_va.copy()
            rng = np.random.default_rng(SEED * 1000 + fold * 10 + k)
            for col in shifted:
                te_cols, va_orig_nan, enc = col_info[col]
                hit = (~va_orig_nan) & (rng.random(len(Xk)) < rates[col])
                idx = np.flatnonzero(hit)
                if col in num_cols:
                    Xk.loc[idx, col] = np.nan
                else:
                    Xk.loc[idx, col] = "missing"
                Xk.loc[idx, te_cols] = enc
            acc += model.predict_proba(Xk)
        oof_tta[val_idx] = acc / K
        f_plain = balanced_accuracy_score(y[val_idx], oof_plain[val_idx].argmax(1))
        f_tta = balanced_accuracy_score(y[val_idx], oof_tta[val_idx].argmax(1))
        print(f"fold {fold}: argmax plain {f_plain:.4f}  tta {f_tta:.4f}")
        del model, X_tr, X_va

    print("\n=== paired OOF read (each arm gets its own decision weights) ===")
    for name, oof in (("plain", oof_plain), ("tta  ", oof_tta)):
        w = tune_decision_weights(y, oof, n_restarts=3)
        pred = weighted_predict(oof, w)
        overall = balanced_accuracy_score(y, pred)
        miss = balanced_accuracy_score(y[orig_nan_any], pred[orig_nan_any])
        comp = balanced_accuracy_score(y[~orig_nan_any], pred[~orig_nan_any])
        am = balanced_accuracy_score(y, oof.argmax(1))
        print(
            f"{name}: dw {overall:.5f}  argmax {am:.5f}  "
            f"miss4 {miss:.5f}  complete4 {comp:.5f}"
        )
    # mixture arm: does a little TTA help even if full replacement doesn't?
    for wmix in (0.25, 0.5):
        mix = (1 - wmix) * oof_plain + wmix * oof_tta
        w = tune_decision_weights(y, mix, n_restarts=3)
        s = balanced_accuracy_score(y, weighted_predict(mix, w))
        print(f"mix w={wmix}: dw {s:.5f}")


if __name__ == "__main__":
    main()
