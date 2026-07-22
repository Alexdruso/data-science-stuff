"""Confident-test pseudo-label canary (Day-18 lens C, C1) — transductive cell.

Never measured on s6e7: distillation-to-student (Day-9) REPLACED the model
and external-50k augmentation (Day-3) trained on OFF-manifold rows — this
cell keeps the model and augments its train folds with ON-manifold test rows.

Design (pre-registered): label test rows where the deployed u4 stack's max
plain proba >= TAU (0.99) with the stack's argmax; append them to each fold's
TRAIN split only (OOF stays honest on real train rows). TE encoder fit stays
on REAL fold-train rows (pseudo rows encoded via the fold's full-map
transform, like test rows) so the TE surface is identical to the twin and the
read isolates the model-training-data channel. Paired 5-fold vs the recorded
twin hgbc_te3_u_s42; instruments argmax + robust-wt refit. Screen bar +0.0007;
escalation = TAU sweep + realmlp consumer + 12-cell seat.

Run: python src/probe_pseudo_hgbc.py -> results/probe_pseudo_hgbc.txt
Resumable per fold via results/cache/ckpt_pseudo/.
"""

from __future__ import annotations

import os
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
TAU = 0.99
TWIN = "hgbc_te3_u_s42"
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
# S6E7_PSEUDO_MATCH=1 (deconfound arm): weight pseudo rows per class so the
# class SHARES of total training mass equal the real fold's — isolates the
# "extra on-manifold examples" channel from the prior shift that
# class_weight="balanced" otherwise re-derives from the augmented mix.
MATCH = os.environ.get("S6E7_PSEUDO_MATCH") == "1"
CKPT = CACHE_DIR / ("ckpt_pseudo_match" if MATCH else "ckpt_pseudo")


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    stack = np.load(RESULTS_DIR / "test_lrstack_u4_r.npy")
    conf = stack.max(axis=1)
    sel = conf >= TAU
    p_lab = stack.argmax(axis=1)
    md_test = ds.test[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    say(f"pseudo pool: tau={TAU} -> {sel.sum()} rows "
        f"({100 * sel.mean():.1f}% of test); class dist "
        f"{np.bincount(p_lab[sel], minlength=N_CLASSES).tolist()}; "
        f"missing-driver share {md_test[sel].mean():.3f} (base {md_test.mean():.3f})")

    # exact trainer preprocessing (train_hgbc_te3, u surface)
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
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
            axis=1,
        )

    X_ps = X_test[sel]
    y_ps = p_lab[sel]
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
        te_ps = enc.transform(te_te_view[sel])  # full-map, like test rows
        X_tr = pd.concat(
            [assemble(X.iloc[tr_idx], te_tr), assemble(X_ps, te_ps)],
            ignore_index=True,
        )
        y_tr = np.concatenate([ds.y[tr_idx], y_ps])
        if MATCH:
            # explicit balanced weights on the REAL fold (== twin's effective
            # weighting; class_weight="balanced" would recompute from the
            # augmented unweighted counts and re-shift the prior). Pseudo rows
            # add mass in the real fold's class shares, so every class scales
            # by the same (1 + ps_mass) factor.
            n_real = np.bincount(ds.y[tr_idx], minlength=N_CLASSES).astype(float)
            n_ps = np.bincount(y_ps, minlength=N_CLASSES).astype(float)
            bw = len(tr_idx) / (N_CLASSES * n_real)
            ps_mass = len(y_ps) / len(tr_idx)
            w_c = np.divide(bw * ps_mass * n_real, n_ps, out=np.zeros(N_CLASSES),
                            where=n_ps > 0)
            sw = np.concatenate([bw[ds.y[tr_idx]], w_c[y_ps]])
            cfg = {**HGBC_CONFIG, "class_weight": None}
            model = HistGradientBoostingClassifier(**cfg, random_state=SEED)
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
            model.fit(X_tr, y_tr)
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
    for name, p in (("twin", twin), ("pseudo", oof)):
        w = robust_decision_weights(ds.y, p)
        say(f"{name:7s} argmax {balanced_accuracy_score(ds.y, p.argmax(1)):.5f}  "
            f"wt {balanced_accuracy_score(ds.y, weighted_predict(p, w)):.5f}")
    say("screen bar +0.0007 paired; escalation = tau sweep + realmlp + 12-cell seat")
    report = RESULTS_DIR / (
        "probe_pseudo_hgbc_match.txt" if MATCH else "probe_pseudo_hgbc.txt"
    )
    report.write_text("\n".join(lines) + "\n")
    print(f"report -> {report}", flush=True)


if __name__ == "__main__":
    main()
