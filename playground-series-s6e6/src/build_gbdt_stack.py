"""GBDT meta-stacker on [base OOF logits + raw features] — region-aware stacking (S6E6).

Our LR stacker (build_lr_stack, 0.9662) consumes base-model OOF logits ONLY, with a
single weight per (model, class) — a GLOBAL reweighting. It cannot do region-dependent
model trust ("at low redshift lean on the STAR-strong model"). A nonlinear meta-learner
(LightGBM) on [logits | raw features] can: it routes on redshift/colors and picks which
base to trust per feature region — directly targeting our error concentration (the low-z
STAR/GALAXY boundary). Memory's own rule: "if stacking, use a nonlinear meta (GBDT on OOF)."

Honesty: the base OOF logits are out-of-fold (clean). The meta is fit/scored via an outer
5-fold on those OOF rows (multi-seed), so its OOF argmax is unbiased. Meta capacity is
deliberately REGULARIZED (small leaves, high min_child_samples, fixed rounds — NO early-stop
on the scored fold) to limit OOF overfit, the known failure mode of GBDT metas.

GATE: OOF argmax must clear ~0.9670 (LR stack 0.9662 + noise floor ~0.0014) to be real.

Run:  python src/build_gbdt_stack.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte", "catboost_v3", "xgb_v3fe",
          "chain_cascade", "chain_cascade_xgb"]
SEEDS = [2024, 7, 13, 42, 99]
EPS = 1e-6
RUN = "gbdtstack"


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    # stacked base logits
    Ztr, Zte, used = [], [], []
    for m in MODELS:
        fo, ft = RESULTS_DIR / f"oof_{m}.npy", RESULTS_DIR / f"test_{m}.npy"
        if fo.exists() and ft.exists():
            Ztr.append(logit(np.load(fo)))
            Zte.append(logit(np.load(ft)))
            used.append(m)
    print(f"Stacking {len(used)} models: {used}")
    logit_cols = [f"{m}_{c}" for m in used for c in ["g", "q", "s"]]
    Ztr_df = pd.DataFrame(np.hstack(Ztr), columns=logit_cols)
    Zte_df = pd.DataFrame(np.hstack(Zte), columns=logit_cols)

    # raw features (the region drivers the LR stacker can't use)
    feat_cols = [c for c in tr.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in feat_cols if tr[c].dtype == pl.String]
    tr_feat, te_feat = tr.select(feat_cols).to_pandas(), te.select(feat_cols).to_pandas()
    for c in cat_cols:
        tr_feat[c] = tr_feat[c].astype("category")
        te_feat[c] = te_feat[c].astype("category")
    X = pd.concat([Ztr_df, tr_feat], axis=1)
    X_test = pd.concat([Zte_df, te_feat], axis=1)
    print(f"meta-features: {X.shape[1]} ({len(logit_cols)} logits + {len(feat_cols)} raw)")

    params = dict(
        objective="multiclass", num_class=3, class_weight="balanced",
        n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=n_jobs, device_type=dev_type,
    )

    oof = np.zeros((len(y), 3))
    test = np.zeros((len(test_ids), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(X, y):
            m = LGBMClassifier(**{**params, "random_state": seed})
            m.fit(X.iloc[tri], y[tri], categorical_feature=cat_cols)
            oof[vai] += m.predict_proba(X.iloc[vai]) / len(SEEDS)
            test += m.predict_proba(X_test) / (len(SEEDS) * 5)

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"GBDT-stack OOF argmax: {argmax:.4f}  [LR-stack 0.9662; GATE clear 0.9670]")
    print(f"GBDT-stack OOF threshold-tuned: {best:.4f}  [LR-stack 0.9663]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  [LR-stack GAL .954/QSO .975/STAR .969]")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, [], best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test)
    labels = le.inverse_transform(np.argmax(test * tw, axis=1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
