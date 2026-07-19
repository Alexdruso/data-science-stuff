"""TE-generation design sweep (Day-16 brainstorm P1) — encoder HPs of the
load-bearing te3/te13 block, never varied (frozen `smooth="auto", cv=5` by
convention; Day-12 tuned only the CONSUMER config over cached blocks).

Arms (each a full seed-42 5-fold hgbc_te3 rerun with one encoder change):
  smooth5 / smooth100 / smooth500 — fixed smoothing vs the EB-auto incumbent
  cv10                            — inner cross-fit 5 -> 10 (shrinks the
                                    train/val TE-distribution gap)
Paired read vs the recorded twin hgbc_te3_u_s42 (OOF argmax 0.95026, weighted
0.950255); incumbent scores recomputed from the on-disk OOF with the same
instrument. Sub-lever NOT here: test-transform bagging changes only test-side
predictions -> OOF-invisible, unreadable offline (recorded as such).

Screen bar +0.0007 (house convention for 1-seed canaries); escalation = the
standard realmlp consumer + 12-cell seat read.

Run: python src/probe_te_design.py -> results/probe_te_design.txt
Resumable per (arm, fold) via results/cache/ckpt_tedesign_*/.
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
ARMS: dict[str, dict[str, object]] = {
    "smooth5": {"smooth": 5.0, "cv": 5},
    "smooth100": {"smooth": 100.0, "cv": 5},
    "smooth500": {"smooth": 500.0, "cv": 5},
    "cv10": {"smooth": "auto", "cv": 10},
}


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(ds.test[col].fillna("missing").astype(str).unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
        )

    def run_arm(arm: str, smooth: object, cv: int) -> np.ndarray:
        ckpt = CACHE_DIR / f"ckpt_tedesign_{arm}"
        ckpt.mkdir(parents=True, exist_ok=True)
        oof = np.zeros((len(ds.y), N_CLASSES))
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
            vck = ckpt / f"f{fold}_val.npy"
            if vck.exists():
                oof[val_idx] = np.load(vck)
                print(f"  [{arm}] fold {fold}: loaded checkpoint", flush=True)
                continue
            enc = TargetEncoder(cv=cv, smooth=smooth, target_type="multiclass")
            te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
            te_va = enc.transform(te_tr_view.iloc[val_idx])
            model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
            model.fit(assemble(X.iloc[tr_idx], te_tr), ds.y[tr_idx])
            val_proba = model.predict_proba(assemble(X.iloc[val_idx], te_va))
            np.save(vck, val_proba)
            oof[val_idx] = val_proba
            score = balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
            print(f"  [{arm}] fold {fold} argmax {score:.4f}", flush=True)
        return oof

    def scores(oof: np.ndarray) -> tuple[float, float]:
        am = float(balanced_accuracy_score(ds.y, oof.argmax(axis=1)))
        w = robust_decision_weights(ds.y, oof)
        wt = float(balanced_accuracy_score(ds.y, weighted_predict(oof, w)))
        return am, wt

    lines = ["TE-generation design sweep — hgbc_te3 canary, seed 42, u surface"]
    inc_oof = np.load(RESULTS_DIR / "oof_hgbc_te3_u_s42.npy")
    inc_am, inc_wt = scores(inc_oof)
    lines.append(f"  incumbent(auto,cv5)  argmax {inc_am:.5f}  weighted {inc_wt:.5f}")
    print(lines[-1], flush=True)
    for arm, cfg in ARMS.items():
        print(f"\n[arm {arm}]", flush=True)
        oof = run_arm(arm, cfg["smooth"], int(cfg["cv"]))  # type: ignore[arg-type]
        am, wt = scores(oof)
        line = (
            f"  {arm:20s} argmax {am:.5f} ({am - inc_am:+.5f})  "
            f"weighted {wt:.5f} ({wt - inc_wt:+.5f})"
        )
        lines.append(line)
        print(line, flush=True)
    lines.append("note: test-transform-bagging sub-lever is OOF-invisible; not")
    lines.append("readable offline — excluded by design, not measured-flat.")
    (RESULTS_DIR / "probe_te_design.txt").write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {RESULTS_DIR / 'probe_te_design.txt'}", flush=True)


if __name__ == "__main__":
    main()
