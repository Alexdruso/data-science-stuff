"""Referee v2 (Day-17): importance-weight the 50k board toward the synthetic joint.

Day-16 verdict: the referee board's non-null ranking (realmlp solo 0.90782 >
champion 0.89696 > stack 0.89339) is JOINT-SHIFT-dominated — the generator
added cross-column dependence (|corr| 0.003 -> 0.10), so every model is scored
off its training joint. This probe deconfounds BY INTERVENTION: reweight the
50k rows by w(x) = P(synth|x)/(1-P(synth|x)) (cross-fitted HGBC discriminator,
original vs COMPLETE-DRIVER synthetic rows so missingness is not the tell;
weights clipped at q0.995, ESS reported) and re-score the board with weighted
balanced accuracy.

Read: if the gaps collapse under reweighting, the joint-shift diagnosis is
confirmed by intervention AND the deconfounded board becomes the usable
independent-noise referee for the finals #2 menu; a candidate still trailing
materially = a surviving flip-noise-overfit flag.

Run: python src/probe_original_referee_v2.py
  -> results/probe_original_referee_v2.txt
Requires the Day-16 referee caches (refprobe_*/f*_ref.npy, refprobe_ustack_*).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES
from lb_anchor import correct
from probe_original_referee import champion_v1_ref, load_referee
from probe_original_referee_deconf import leg_ref
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    load_dataset,
    robust_decision_weights,
)
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle_utils import weighted_predict

SEED = 42


def weighted_ba(
    y: NDArray[np.int64], pred: NDArray[np.int64], w: NDArray[np.float64]
) -> float:
    recalls = []
    for c in range(N_CLASSES):
        m = y == c
        denom = w[m].sum()
        recalls.append(float((w[m] * (pred[m] == c)).sum() / denom))
    return float(np.mean(recalls))


def main() -> None:
    ds = load_dataset()
    X_ref, y_ref = load_referee(ds)
    feat_cols = list(ds.train.columns)

    # discriminator frame: original 50k (1) vs complete-driver synthetic rows (0)
    complete = ds.train[feat_cols].notna().all(axis=1).to_numpy()
    Xs = ds.train.loc[complete, feat_cols]
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xs), size=min(150_000, len(Xs)), replace=False)
    Xs = Xs.iloc[sub]
    # shared categorical coding across both frames
    import pandas as pd

    both = pd.concat([Xs, X_ref[feat_cols]], axis=0, ignore_index=True)
    for col in ds.cat_cols:
        both[col] = both[col].astype("category").cat.codes
    Xd = both.to_numpy(dtype=np.float64)
    yd = np.concatenate([np.zeros(len(Xs)), np.ones(len(X_ref))]).astype(int)

    # cross-fitted P(orig|x) for every row
    p_orig = np.zeros(len(yd))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(Xd, yd):
        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(Xd[tr], yd[tr])
        p_orig[te] = clf.predict_proba(Xd[te])[:, 1]
    auc = roc_auc_score(yd, p_orig)
    p_ref = p_orig[len(Xs):]
    # weight referee rows toward the SYNTHETIC joint: w = P(synth|x)/P(orig|x)
    w = (1.0 - p_ref) / np.clip(p_ref, 1e-6, None)
    w = np.clip(w, None, np.quantile(w, 0.995))
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w**2).sum())

    legs = {m: leg_ref(m) for m in ["lgbm", "xgboost", "catboost", "realmlp_te13"]}
    champ = champion_v1_ref(ds.y, legs)
    s_ref = np.load(CACHE_DIR / "refprobe_ustack_sref.npy")
    soof = np.load(CACHE_DIR / "refprobe_ustack_soof.npy")
    with (RESULTS_DIR / "decision_weights_realmlp_te13_u_breadth.json").open() as f:
        rw_map = json.load(f)
    rw = np.array([rw_map[c] for c in CLASSES])
    preds = {
        "champion_v1": champ.argmax(axis=1),
        "lrstack_u4(6-leg)": weighted_predict(s_ref, robust_decision_weights(ds.y, soof)),
        "realmlp_te13 solo": correct(legs["realmlp_te13"], rw).argmax(axis=1),
    }

    lines = [
        "referee v2 — importance-weighted toward the synthetic joint",
        f"discriminator AUC {auc:.4f} (orig vs complete-driver synth); "
        f"ESS {ess:,.0f} / {len(w):,} referee rows",
        "",
        f"{'candidate':22s} {'unweighted':>11s} {'iw-toward-synth':>16s} {'delta':>9s}",
    ]
    uw = {}
    for name, pred in preds.items():
        ones = np.ones(len(y_ref))
        b_u = weighted_ba(y_ref, np.asarray(pred), ones)
        b_w = weighted_ba(y_ref, np.asarray(pred), w)
        uw[name] = (b_u, b_w)
        lines.append(f"{name:22s} {b_u:11.5f} {b_w:16.5f} {b_w - b_u:+9.5f}")
    gap_u = uw["realmlp_te13 solo"][0] - uw["lrstack_u4(6-leg)"][0]
    gap_w = uw["realmlp_te13 solo"][1] - uw["lrstack_u4(6-leg)"][1]
    lines.append("")
    lines.append(
        f"solo-minus-stack gap: unweighted {gap_u:+.5f} -> weighted {gap_w:+.5f} "
        f"(collapse => joint-shift confirmed by intervention)"
    )
    if ess < 1000:
        lines.append(f"⚠️ ESS {ess:,.0f} < 1,000 — weighted read is underpowered")
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_original_referee_v2.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
