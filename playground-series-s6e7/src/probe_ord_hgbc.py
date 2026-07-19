"""Cumulative-link (ordinal) target formulation — CPU canary (Day-16 brainstorm M1).

The one unmeasured target-reparametrization cell: the Day-11 idea screen held
no ordinal item, and the Day-9 cascade kill was a DIFFERENT factorization
(independent binary stages), not nested thresholds on a shared severity latent
fit < at-risk < unhealthy. Mechanism claim: the generator thresholds a latent
at the sleep 6/7 sigmoid cuts; a cumulative link pools strength across both
thresholds and enforces unimodality along the severity axis — a boundary shape
no softmax leg expresses.

Canary: two cumulative HGBC binaries on IDENTICAL te3 inputs (same multiclass
TargetEncoder, same tuned config — only the head/objective changes, so the
read is paired): b1 = P(o>=1) = P(not fit), b2 = P(o>=2) = P(unhealthy).
Readout (difference form): P(fit)=1-b1, P(at-risk)=b1-b2, P(unhealthy)=b2,
clipped at 0 and renormalized. CLASSES order [at-risk, fit, unhealthy] is
alphabetical — at-risk is ordinal MIDDLE; columns are emitted in CLASSES order.

Paired read vs recorded twin hgbc_te3_u_s42 (argmax 0.95026 / weighted
0.950255). Escalation gate +0.0004 paired -> realmlp CORN head + seat read.

Run: python src/probe_ord_hgbc.py -> results/probe_ord_hgbc.txt
Resumable per fold via results/cache/ckpt_ordhgbc/.
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
# ordinal severity: fit(cls 1) < at-risk(cls 0) < unhealthy(cls 2)
ORD_OF_CLS = {1: 0, 0: 1, 2: 2}


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    o = np.vectorize(ORD_OF_CLS.get)(ds.y)
    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        cats = sorted(
            set(X[col].unique())
            | set(ds.test[col].fillna("missing").astype(str).unique())
        )
        X[col] = pd.Categorical(X[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
        )

    cfg = {k: v for k, v in HGBC_CONFIG.items() if k != "class_weight"}
    ckpt = CACHE_DIR / "ckpt_ordhgbc"
    ckpt.mkdir(parents=True, exist_ok=True)
    oof = np.zeros((len(ds.y), N_CLASSES))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        vck = ckpt / f"f{fold}_val.npy"
        if vck.exists():
            oof[val_idx] = np.load(vck)
            print(f"  fold {fold}: loaded checkpoint", flush=True)
            continue
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        cum = np.zeros((len(val_idx), 2))
        for j, thresh in enumerate((1, 2)):
            b = (o[tr_idx] >= thresh).astype(np.int64)
            model = HistGradientBoostingClassifier(
                **cfg, class_weight="balanced", random_state=SEED
            )
            model.fit(X_tr, b)
            cum[:, j] = model.predict_proba(X_va)[:, 1]
        # monotone repair (cum must be non-increasing) then difference readout
        cum[:, 1] = np.minimum(cum[:, 1], cum[:, 0])
        p_ord = np.stack(
            [1.0 - cum[:, 0], cum[:, 0] - cum[:, 1], cum[:, 1]], axis=1
        )  # ordinal order: fit, at-risk, unhealthy
        p_cls = np.clip(p_ord[:, [1, 0, 2]], 1e-9, None)  # back to CLASSES order
        val_proba = p_cls / p_cls.sum(axis=1, keepdims=True)
        np.save(vck, val_proba)
        oof[val_idx] = val_proba
        score = balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
        print(f"  fold {fold} argmax {score:.4f}", flush=True)

    def scores(p: np.ndarray) -> tuple[float, float]:
        am = float(balanced_accuracy_score(ds.y, p.argmax(axis=1)))
        w = robust_decision_weights(ds.y, p)
        wt = float(balanced_accuracy_score(ds.y, weighted_predict(p, w)))
        return am, wt

    inc_oof = np.load(RESULTS_DIR / "oof_hgbc_te3_u_s42.npy")
    inc_am, inc_wt = scores(inc_oof)
    am, wt = scores(oof)
    err_inc = inc_oof.argmax(axis=1) != ds.y
    err_ord = oof.argmax(axis=1) != ds.y
    overlap = float((err_inc & err_ord).sum() / max(err_ord.sum(), 1))
    lines = [
        "cumulative-link ordinal HGBC canary — seed 42, u surface, te3 inputs",
        f"  incumbent softmax  argmax {inc_am:.5f}  weighted {inc_wt:.5f}",
        f"  ordinal cumlink    argmax {am:.5f} ({am - inc_am:+.5f})  "
        f"weighted {wt:.5f} ({wt - inc_wt:+.5f})",
        f"  error overlap (ord errors also incumbent errors): {overlap:.3f}",
        "escalation gate: paired weighted >= +0.0004 -> realmlp CORN + seat read",
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_ord_hgbc.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
