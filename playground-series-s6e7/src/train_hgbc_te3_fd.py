"""Frequency-distortion features (Day-10, GF #4 — pre-check PASSED).

probe_freq_distortion measured class-conditional per-value frequency
distortion synth-vs-original, decisively on the water_intake control (23/118
FDR-sig values) — a CTGAN conditional-vector fingerprint. TE sees only
P_synth(y|v); the distortion RATIO r(v) = P_synth(v)/P_orig(v) is a different
statistic that pools across values ("oversampled values lean class c").

Features (label-free, fold-independent — marginal counts only): per-column
log r(v) with Laplace smoothing (synth = train+test pooled; values unseen in
the original get the capped max ratio) + the row's total |log r| distortion
sum. 14 columns appended to the hgbc_te3 design, paired vs hgbc_te3_u_s42
(0.9504).

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_fd.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import DATA_DIR, N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"
ALPHA = 0.5
CAP = 6.0


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ds = load_dataset()
    raw_cols = ds.feature_cols
    ext = pd.read_csv(EXT_CSV)

    fd_names = [f"fd_{c}" for c in raw_cols] + ["fd_sum"]
    fd_tr = np.zeros((len(ds.train), len(raw_cols)), dtype=np.float32)
    fd_te = np.zeros((len(ds.test), len(raw_cols)), dtype=np.float32)
    for j, c in enumerate(raw_cols):
        synth = pd.concat([ds.train[c], ds.test[c]]).astype(str)
        n_s = synth.value_counts()
        n_o = ext[c].astype(str).value_counts()
        vals = n_s.index
        p_s = (n_s + ALPHA) / (n_s.sum() + ALPHA * len(vals))
        p_o = (n_o.reindex(vals).fillna(0) + ALPHA) / (n_o.sum() + ALPHA * len(vals))
        logr = np.log(p_s / p_o).clip(-CAP, CAP)
        fd_tr[:, j] = ds.train[c].astype(str).map(logr).to_numpy(dtype=np.float32)
        fd_te[:, j] = ds.test[c].astype(str).map(logr).to_numpy(dtype=np.float32)
    fd_tr = np.hstack([fd_tr, np.abs(fd_tr).sum(axis=1, keepdims=True)])
    fd_te = np.hstack([fd_te, np.abs(fd_te).sum(axis=1, keepdims=True)])
    print(f"freq-distortion cols: {len(fd_names)}; "
          f"train |logr| mean {np.abs(fd_tr[:, :-1]).mean():.3f}")

    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    X[fd_names] = fd_tr
    X_test[fd_names] = fd_te
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te3fd_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3fd", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3fd_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
