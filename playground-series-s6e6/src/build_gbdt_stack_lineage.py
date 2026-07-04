"""GBDT meta-stacker A/B: does adding fold-safe ALPHA-LINEAGE rates to the meta help? (S6E6)

The alpha-lineage signal (per-value class rate of the synthetic generator's reused `alpha`
floats) is REAL — a perfect-gating oracle clears the noise floor (+0.0029, 91% GALAXY rescues)
— but a strong XGB base IGNORED it (importance ~0): bolted among 240 photometry features, the
booster never trusts the rate over a confident photometric split. The right consumer is the
REGION-AWARE meta-stacker, which sees the base LOGITS: it can learn "when the bases are only
mildly confident AND the alpha-group leans GALAXY, nudge GALAXY" — exactly the rescue the
oracle identified.

This runs the meta BOTH ways (no-lineage vs +lineage) on identical folds, so the delta is
clean. Lineage rates are fold-safe (TargetEncoder fit on the meta fold's train only) and
restricted to test-present alphas (kills train-only-group inflation → OOF predicts LB). Prints
only; does not overwrite the production gbdtstack files.

GATE: +lineage OOF thresh must clear no-lineage by > ~0.0014 (noise floor).

Run:  python src/build_gbdt_stack_lineage.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte", "catboost_v3", "xgb_v3fe",
          "chain_cascade", "chain_cascade_xgb"]
SEEDS = [2024, 7, 13, 42, 99]
EPS = 1e-6
RARE = "__rare__"
LINEAGE_SMOOTH = 2.0
CLASSES = ["GALAXY", "QSO", "STAR"]
LINEAGE_COLS = [f"alpha_rate_{c}" for c in CLASSES] + ["alpha_log_groupsize"]


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())

    Ztr, used = [], []
    for m in MODELS:
        fo = RESULTS_DIR / f"oof_{m}.npy"
        if fo.exists():
            Ztr.append(logit(np.load(fo)))
            used.append(m)
    print(f"Stacking {len(used)} models: {used}", flush=True)
    logit_cols = [f"{m}_{c}" for m in used for c in ["g", "q", "s"]]
    Ztr_df = pd.DataFrame(np.hstack(Ztr), columns=logit_cols)

    feat_cols = [c for c in tr.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in feat_cols if tr[c].dtype == pl.String]
    tr_feat = tr.select(feat_cols).to_pandas()
    for c in cat_cols:
        tr_feat[c] = tr_feat[c].astype("category")
    X = pd.concat([Ztr_df, tr_feat], axis=1)

    # lineage pieces (id-sorted, aligned to oof arrays). group-size target-free → precompute.
    alpha = tr["alpha"].to_numpy()
    test_alphas = set(te["alpha"].to_list())
    cnt: Counter = Counter(alpha.tolist())
    cnt.update(te["alpha"].to_list())
    gsize = np.log1p([cnt[a] for a in alpha]).astype(np.float64)
    key = np.array([str(a) if a in test_alphas else RARE for a in alpha]).reshape(-1, 1)
    print(f"meta-features: {X.shape[1]} (+{len(LINEAGE_COLS)} lineage in the +lineage arm)",
          flush=True)

    params = dict(
        objective="multiclass", num_class=3, class_weight="balanced",
        n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        verbose=-1, n_jobs=n_jobs, device_type=dev_type,
    )

    oof_base = np.zeros((len(y), 3))
    oof_lin = np.zeros((len(y), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(X, y):
            # fold-safe lineage rates: one TE fit on this fold's train
            enc = TargetEncoder(target_type="multiclass", smooth=LINEAGE_SMOOTH, cv=5,
                                random_state=42)
            enc.fit(key[tri], y[tri])
            rate_tr = enc.transform(key[tri])
            rate_va = enc.transform(key[vai])

            Xtr, Xva = X.iloc[tri], X.iloc[vai]
            mb = LGBMClassifier(**{**params, "random_state": seed})
            mb.fit(Xtr, y[tri], categorical_feature=cat_cols)
            oof_base[vai] += mb.predict_proba(Xva) / len(SEEDS)

            Xtr_l = Xtr.copy()
            Xva_l = Xva.copy()
            for k, col in enumerate(CLASSES):
                Xtr_l[f"alpha_rate_{col}"] = rate_tr[:, k]
                Xva_l[f"alpha_rate_{col}"] = rate_va[:, k]
            Xtr_l["alpha_log_groupsize"] = gsize[tri]
            Xva_l["alpha_log_groupsize"] = gsize[vai]
            ml = LGBMClassifier(**{**params, "random_state": seed})
            ml.fit(Xtr_l, y[tri], categorical_feature=cat_cols)
            oof_lin[vai] += ml.predict_proba(Xva_l) / len(SEEDS)
        print(f"  seed {seed} done", flush=True)

    for name, oof in [("no-lineage", oof_base), ("+lineage", oof_lin)]:
        pred = np.argmax(oof, axis=1)
        argmax = float(balanced_accuracy_score(y, pred))
        rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
        _, best = optimize_thresholds(oof, y)
        print(f"[{name:>11}] argmax {argmax:.6f}  thresh {best:.6f}  "
              f"recall G/Q/S {rec[0]:.4f}/{rec[1]:.4f}/{rec[2]:.4f}")
    np.save(RESULTS_DIR / "oof_gbdtstack_lineage.npy", oof_lin)
    print("saved oof_gbdtstack_lineage.npy")


if __name__ == "__main__":
    main()
