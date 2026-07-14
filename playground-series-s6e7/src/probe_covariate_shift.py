"""Covariate-shift importance-weighting probe (Day-11, advisor-directed).

AV AUC=0.65 (measured, CLAUDE.md) says train/test differ, but the campaign never
ACTED on it. If the shift is pure covariate (P(x) moves, P(y|x) fixed) the Bayes
predictor is unchanged and weighting is a no-op. If there is CONCEPT shift (the flip
rule itself differs in the test-like region) then importance-weighting the training
loss toward the test distribution sharpens the posterior where it counts — a lever no
te13 recombination can reach (orthogonal to the stress x activity x sleep channel).

Steps:
  1. Adversarial classifier: HGBC on raw features predicting is_test, 5-fold OOF over
     concat(train,test) -> leak-free P(test|x) for every train row.
  2. Weights w = clip(p/(1-p), [0.2, 5]) — density ratio toward the test distribution.
  3. Train the exact hgbc_te3 pipeline UNWEIGHTED and IMPORTANCE-WEIGHTED (sample_weight
     on HGBC.fit; TE stays unweighted — no weighted API).
  4. Report plain OOF BA and AV-WEIGHTED OOF BA for both. Lever iff weighted model's
     AV-weighted BA beats unweighted by >= +0.001.

Run: python src/probe_covariate_shift.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score as ba
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, load_dataset  # noqa: E402
from train_hgbc_te3 import HGBC_CONFIG  # noqa: E402
from zoo_common import zoo_cv  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def av_weights(X: pd.DataFrame, X_test: pd.DataFrame, cat_cols) -> np.ndarray:  # noqa: ANN001
    n_tr = len(X)
    both = pd.concat([X, X_test], ignore_index=True)
    for c in cat_cols:
        both[c] = pd.Categorical(both[c].fillna("missing").astype(str))
    is_test = np.r_[np.zeros(n_tr), np.ones(len(X_test))].astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
        categorical_features="from_dtype", early_stopping=True, random_state=42,
    )
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    p = cross_val_predict(clf, both, is_test, cv=skf, method="predict_proba")[:, 1]
    print(f"AV AUC = {roc_auc_score(is_test, p):.4f}")
    p_tr = np.clip(p[:n_tr], 1e-4, 1 - 1e-4)
    w = np.clip(p_tr / (1 - p_tr), 0.2, 5.0)
    w = w * (len(w) / w.sum())  # normalize to mean 1
    print(f"train weights: min {w.min():.3f} med {np.median(w):.3f} "
          f"max {w.max():.3f} | frac>1.5: {(w > 1.5).mean():.3f}")
    return w


def main() -> None:
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    w_all = av_weights(X, X_test, ds.cat_cols)

    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def make_fold(weighted: bool):  # noqa: ANN202
        def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
            enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
            te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
            te_va = enc.transform(te_tr_view.iloc[val_idx])
            te_te = enc.transform(te_te_view)

            def asm(base, blk):  # noqa: ANN001, ANN202
                return pd.concat(
                    [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                    axis=1,
                )

            model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
            sw = w_all[tr_idx] if weighted else None
            model.fit(asm(X.iloc[tr_idx], te_tr), ds.y[tr_idx], sample_weight=sw)
            return model.predict_proba(asm(X.iloc[val_idx], te_va)), \
                model.predict_proba(asm(X_test, te_te))

        return fit_fold

    y = ds.y
    print("\n=== UNWEIGHTED ===", flush=True)
    oof_u, _, _ = zoo_cv(ds, make_fold(False), ckpt_name="cs_unw", seed=SEEDS[0])
    print("=== IMPORTANCE-WEIGHTED ===", flush=True)
    oof_w, _, _ = zoo_cv(ds, make_fold(True), ckpt_name="cs_iw", seed=SEEDS[0])

    print("\n---- results (higher = better) ----")
    for name, oof in (("unweighted", oof_u), ("weighted", oof_w)):
        plain = ba(y, oof.argmax(1))
        wtd = ba(y, oof.argmax(1), sample_weight=w_all)
        print(f"  {name:11s}  plain BA {plain:.4f}   AV-weighted BA {wtd:.4f}")
    print(f"\n  Δ AV-weighted BA (weighted - unweighted) = "
          f"{ba(y, oof_w.argmax(1), sample_weight=w_all) - ba(y, oof_u.argmax(1), sample_weight=w_all):+.4f}")


if __name__ == "__main__":
    main()
