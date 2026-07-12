"""Per-class specialist binaries as META COLUMNS (Day-10 eve, TE-eng #5).

Two binary LGBMs on the te3 recipe — fit-vs-rest and unhealthy-vs-rest — whose
OOF/test probabilities join the LR-stack DESIGN as 2 extra columns. This adds
INPUTS to the meta (a different statistic: binary-target TE tables + a binary
boundary the multiclass objective never optimizes directly), NOT the dead
cascade decision structure (which recombined probabilities). Folds identical
to the zoo (seed 42) so the columns are honestly cross-fitted for the meta.

Artifacts: results/oof_bins_te3_u_s42.npy / test_bins_te3_u_s42.npy — (n, 2)
columns [P(fit), P(unhealthy)]. Seat test consumes them as logit columns.

Run: S6E7_SEEDS=42 python src/train_binaries_te3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from lgbm_device import get_lgbm_device
from train_common import RESULTS_DIR, SEEDS, load_dataset
from train_lgbm_te3 import LGBM_TE3_PARAMS
from zoo_common import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

TARGETS = {"fit": 1, "unh": 2}


def main() -> None:
    ds = load_dataset()
    seed = SEEDS[0]
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}" for c in raw_cols]

    params = {
        **{k: v for k, v in LGBM_TE3_PARAMS.items() if k not in ("num_class",)},
        "objective": "binary",
        "metric": "binary_logloss",
    }

    oof = np.zeros((len(ds.y), len(TARGETS)))
    test = np.zeros((len(ds.test_ids), len(TARGETS)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ckpt = CACHE_DIR / f"ckpt_bins_te3_u_s{seed}"
    ckpt.mkdir(parents=True, exist_ok=True)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        vck, tck = ckpt / f"f{fold}_val.npy", ckpt / f"f{fold}_test.npy"
        if vck.exists() and tck.exists():
            oof[val_idx] = np.load(vck)
            test += np.load(tck) / 5
            print(f"fold {fold}: loaded checkpoint")
            continue
        val_out = np.zeros((len(val_idx), len(TARGETS)))
        test_out = np.zeros_like(test)
        for j, (tag, cls) in enumerate(TARGETS.items()):
            yb = (ds.y == cls).astype(np.int64)
            enc = TargetEncoder(cv=5, smooth="auto", target_type="binary")
            te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], yb[tr_idx])
            te_va = enc.transform(te_tr_view.iloc[val_idx])
            te_te = enc.transform(te_te_view)

            def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
                return pd.concat(
                    [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                    axis=1,
                )

            model = LGBMClassifier(**{**params, "random_state": seed})
            model.fit(
                assemble(X.iloc[tr_idx], te_tr),
                yb[tr_idx],
                eval_set=[(assemble(X.iloc[val_idx], te_va), yb[val_idx])],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            val_out[:, j] = model.predict_proba(assemble(X.iloc[val_idx], te_va))[:, 1]
            test_out[:, j] = model.predict_proba(assemble(X_test, te_te))[:, 1]
            print(f"  fold {fold} {tag}: done", flush=True)
        np.save(vck, val_out)
        np.save(tck, test_out * 5)
        oof[val_idx] = val_out
        test += test_out

    np.save(RESULTS_DIR / f"oof_bins_te3_u_s{seed}.npy", oof)
    np.save(RESULTS_DIR / f"test_bins_te3_u_s{seed}.npy", test)
    for j, tag in enumerate(TARGETS):
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score((ds.y == TARGETS[tag]).astype(int), oof[:, j])
        print(f"{tag}-vs-rest OOF AUC: {auc:.5f}")
    print("saved bins_te3 arrays")


if __name__ == "__main__":
    main()
