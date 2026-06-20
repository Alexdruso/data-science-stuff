"""Linear / generative base probe — how strong & how DIVERSE are classical models? (S6E6)

Answers two questions the GBDT/NN cluster can't:
  1. How strong is plain logistic regression on our features? (+ LR with degree-2
     interactions, LDA, QDA — generative Gaussian classifiers with a totally different
     inductive bias from trees/NNs.)
  2. Does any of them DECORRELATE the stack? A linear/generative base makes structurally
     different errors (straight / quadratic boundaries vs axis-aligned splits vs learned
     manifolds). The gold-standard diversity test = add its OOF to the 7-base LR stacker
     and read the argmax delta, NOT standalone score.

5-fold stratified (rs=42), numerics standardized, cats one-hot. class_weight balanced for
LR; LDA/QDA use class priors (we report per-class recall to see the imbalance handling).
Saves oof/test_<name>.npy for any that look worth keeping.

Run:  python src/probe_linear_bases.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
STACK_MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
                "xgb_deotte", "realmlp_deotte", "catboost_deotte"]
NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
            "u_g", "g_r", "r_i", "i_z", "u_z", "log1p_redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def design_matrix(tr: pl.DataFrame, te: pl.DataFrame):
    """Numeric block + one-hot cats, as numpy. Returns (Xtr_num, Xte_num, Xtr_oh, Xte_oh)."""
    num_tr = tr.select(NUM_COLS).to_numpy().astype(np.float64)
    num_te = te.select(NUM_COLS).to_numpy().astype(np.float64)
    oh_tr, oh_te = [], []
    for c in CAT_COLS:
        cats = sorted(set(tr[c].to_list()))
        idx = {v: k for k, v in enumerate(cats)}
        for v in cats:
            oh_tr.append((tr[c].to_numpy() == v).astype(np.float64))
            oh_te.append((te[c].to_numpy() == v).astype(np.float64))
    oh_tr = np.vstack(oh_tr).T if oh_tr else np.empty((len(tr), 0))
    oh_te = np.vstack(oh_te).T if oh_te else np.empty((len(te), 0))
    return num_tr, num_te, oh_tr, oh_te


def make_model(name: str):
    if name == "lr":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    if name == "lr_poly":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=0.01)
    raise ValueError(name)


def fit_oof(name, num_tr, num_te, oh_tr, oh_te, y, n_test):
    """5-fold OOF + test probs for one classical model. Standardizes inside each fold."""
    poly = name == "lr_poly"
    oof = np.zeros((len(y), 3))
    test = np.zeros((n_test, 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tri, vai in skf.split(num_tr, y):
        sc = StandardScaler().fit(num_tr[tri])
        Ntr, Nva, Nte = sc.transform(num_tr[tri]), sc.transform(num_tr[vai]), sc.transform(num_te)
        if poly:
            pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(Ntr)
            Ntr, Nva, Nte = pf.transform(Ntr), pf.transform(Nva), pf.transform(Nte)
        Xtr = np.hstack([Ntr, oh_tr[tri]])
        Xva = np.hstack([Nva, oh_tr[vai]])
        Xte = np.hstack([Nte, oh_te])
        m = make_model(name)
        m.fit(Xtr, y[tri])
        oof[vai] = m.predict_proba(Xva)
        test += m.predict_proba(Xte) / 5
    return oof, test


def stack_delta(extra_oof, extra_test, y, n_test):
    """Argmax of the 7-base LR stack WITH vs WITHOUT the extra base (single seed, fast)."""
    base_tr, base_te = [], []
    for mdl in STACK_MODELS:
        base_tr.append(logit(np.load(RESULTS_DIR / f"oof_{mdl}.npy")))
        base_te.append(logit(np.load(RESULTS_DIR / f"test_{mdl}.npy")))

    def run(blocks_tr):
        Z = np.hstack(blocks_tr)
        oof = np.zeros((len(y), 3))
        skf = StratifiedKFold(5, shuffle=True, random_state=2024)
        for tri, vai in skf.split(Z, y):
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            lr.fit(Z[tri], y[tri])
            oof[vai] = lr.predict_proba(Z[vai])
        return float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))

    base = run(base_tr)
    withx = run(base_tr + [logit(extra_oof)])
    return base, withx


def main() -> None:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    n_test = len(te)
    num_tr, num_te, oh_tr, oh_te = design_matrix(tr, te)
    print(f"design: {num_tr.shape[1]} numeric + {oh_tr.shape[1]} one-hot cat features\n")

    results = {}
    for name in ["lr", "lr_poly", "lda", "qda"]:
        oof, test = fit_oof(name, num_tr, num_te, oh_tr, oh_te, y, n_test)
        am = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
        rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
        base, withx = stack_delta(oof, test, y, n_test)
        results[name] = (am, withx - base)
        print(f"{name:8s} standalone {am:.4f}  "
              f"recall {dict(zip(le.classes_, rec.round(3)))}")
        print(f"{' '*8} stack 7-base {base:.5f} -> +{name} {withx:.5f}  "
              f"(delta {withx-base:+.5f})")
        np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
        np.save(RESULTS_DIR / f"test_{name}.npy", test)
        print()

    print("summary (standalone / stack-delta vs 7-base):")
    for n, (am, d) in results.items():
        print(f"  {n:8s} {am:.4f}   {d:+.5f}")


if __name__ == "__main__":
    main()
