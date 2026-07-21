"""Driver reconstruction probe (Day-17 brainstorm P1, information-pathway lens).

The archaeologist kill-shot conditions on EXACT sleep, so its 3-dim closure
covers complete-driver rows only; the missing-driver regime (26% of rows, 86%
of residual error) is governed by P(label | observed pattern). The generator
ADDED cross-column dependence (orig |corr| 0.003 -> synth ~0.10), so observed
columns carry signal about the missing driver VALUE — and te13 is a set of
per-column posteriors, not a joint map into the rule's own coordinate.

Design (paired, identical instrument — probe_te_design pattern):
  base — hgbc_te3 exactly (inline TE over 13 string cols + raw cols)
  imp  — base + 2 cols: imp_sleep (fold-safe HGBR sleep_duration ~ other 12,
         fit on fold-train rows with observed sleep) and sleep_completed
         (observed value where present, else imp_sleep)
TE encodings are computed once per fold and shared by both arms, so the delta
isolates the reconstruction channel. Screen bar +0.0007 (1-seed canary);
escalation = realmlp consumer + 12-cell seat.

Run: python src/probe_impute_driver.py -> results/probe_impute_driver.txt
Resumable per (arm, fold) via results/cache/ckpt_impdrv_*/.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
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

# S6E7_PROBE_SEED (Day-17): seed-replication arm — the fold split, TE inner cv,
# imputer, and HGBC all re-seed together (exact sibling of the K=10 s7 pattern).
SEED = int(os.environ.get("S6E7_PROBE_SEED", "42"))
DRIVER = "sleep_duration"


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
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
    other_cols = [c for c in raw_cols if c != DRIVER]
    sleep = ds.train[DRIVER].to_numpy(dtype=np.float64)
    obs = ~np.isnan(sleep)
    print(f"driver {DRIVER}: {obs.mean():.3f} observed", flush=True)

    def assemble(base: pd.DataFrame, blk: np.ndarray, extra: "dict[str, np.ndarray] | None" = None) -> pd.DataFrame:
        parts = [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)]
        if extra:
            parts.append(pd.DataFrame(extra))
        return pd.concat(parts, axis=1)

    ckpt = CACHE_DIR / f"ckpt_impdrv_s{SEED}" if SEED != 42 else CACHE_DIR / "ckpt_impdrv"
    ckpt.mkdir(parents=True, exist_ok=True)
    oofs = {"base": np.zeros((len(ds.y), N_CLASSES)), "imp": np.zeros((len(ds.y), N_CLASSES))}
    r2s = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        done = {a: (ckpt / f"{a}_f{fold}_val.npy") for a in oofs}
        if all(p.exists() for p in done.values()):
            for a, p in done.items():
                oofs[a][val_idx] = np.load(p)
            print(f"  fold {fold}: loaded checkpoints", flush=True)
            continue

        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])

        # fold-safe driver reconstruction: fit on fold-train rows with observed sleep
        fit_mask = np.zeros(len(ds.y), dtype=bool)
        fit_mask[tr_idx] = True
        fit_mask &= obs
        reg = HistGradientBoostingRegressor(
            max_iter=300, early_stopping=True, categorical_features="from_dtype",
            random_state=SEED,
        )
        reg.fit(X.loc[fit_mask, other_cols], sleep[fit_mask])
        imp_all = reg.predict(X[other_cols]).astype(np.float64)
        val_obs = np.zeros(len(ds.y), dtype=bool)
        val_obs[val_idx] = True
        val_obs &= obs
        r2 = r2_score(sleep[val_obs], imp_all[val_obs])
        r2s.append(r2)
        print(f"  fold {fold}: imputer val R2 {r2:.4f}", flush=True)
        completed = np.where(np.isnan(sleep), imp_all, sleep)

        for arm in ("base", "imp"):
            if done[arm].exists():
                oofs[arm][val_idx] = np.load(done[arm])
                continue
            extra_tr = extra_va = None
            if arm == "imp":
                extra_tr = {"imp_sleep": imp_all[tr_idx], "sleep_completed": completed[tr_idx]}
                extra_va = {"imp_sleep": imp_all[val_idx], "sleep_completed": completed[val_idx]}
            model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
            model.fit(assemble(X.iloc[tr_idx], te_tr, extra_tr), ds.y[tr_idx])
            val_proba = model.predict_proba(assemble(X.iloc[val_idx], te_va, extra_va))
            np.save(done[arm], val_proba)
            oofs[arm][val_idx] = val_proba
            score = balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
            print(f"  [{arm}] fold {fold} argmax {score:.4f}", flush=True)

    lines = [f"driver reconstruction probe — hgbc_te3 canary, seed {SEED}, u surface"]
    if r2s:
        lines.append(f"imputer val R2 per fold: {' '.join(f'{r:.4f}' for r in r2s)}")
    res = {}
    for arm, oof in oofs.items():
        am = float(balanced_accuracy_score(ds.y, oof.argmax(axis=1)))
        w = robust_decision_weights(ds.y, oof)
        wt = float(balanced_accuracy_score(ds.y, weighted_predict(oof, w)))
        res[arm] = (am, wt)
        lines.append(f"  {arm:5s} argmax {am:.5f}  weighted {wt:.5f}")
    d_am = res["imp"][0] - res["base"][0]
    d_wt = res["imp"][1] - res["base"][1]
    lines.append(f"  paired delta: argmax {d_am:+.5f}  weighted {d_wt:+.5f} (bar +0.0007)")
    report = "\n".join(lines)
    print(report, flush=True)
    out = "probe_impute_driver.txt" if SEED == 42 else f"probe_impute_driver_s{SEED}.txt"
    (RESULTS_DIR / out).write_text(report + "\n")


if __name__ == "__main__":
    main()
