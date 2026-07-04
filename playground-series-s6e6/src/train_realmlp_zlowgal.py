"""RealMLP + one-sided low-z-GALAXY loss boost (S6E6 — redesign of realmlp_zbalw).

realmlp_zbalw (within-class inverse-z-freq weighting) was FLAT/NEGATIVE because inverse
frequency DOWN-weights the 65%-majority GALAXY's bulk (peaked z≈0.51) → typical-GALAXY recall
fell 0.96→0.9457. This fixes the flaw with a strictly ONE-SIDED weight: every row stays at
weight 1 EXCEPT GALAXY rows in STAR's redshift territory (the low-z GALAXY/STAR confusion zone),
which get a fixed boost. Nothing is down-weighted, so GALAXY's bulk is untouched; the only
intervention is extra loss mass on the specific bottleneck rows.

Confusion zone is data-driven & fold-safe: z_thr = the STAR_Q quantile of the TRAIN-fold STAR
redshift (STAR lives at z≈0.07, so its high quantile marks where STAR density fades); GALAXY rows
below z_thr coexist with STAR. Boost is a flat factor (NOT inverse-freq — that was the bug).

Deconfounded vs realmlp_zbalw: keeps AMP(fp16)+fused-AdamW (numerically ~neutral) but DROPS
torch.compile, so dropout runs its real expm4t schedule (compile forced constant dropout). The
only meaningful difference from realmlp_deotte is therefore the weight itself.

GATE: standalone OOF should now land NEAR realmlp_deotte 0.96888 (the zbalw 0.959 drop should
recover — if it doesn't, AMP/weight bug). Want GALAXY recall UP without STAR cratering below
~0.93. Then eval_stack_delta is binding.

Run:  python src/train_realmlp_zlowgal.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from realmlp_deotte import CONFIG, RealMLP_TD_Classifier
from train_realmlp_deotte import feature_engineering

warnings.filterwarnings("ignore")
from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
CKPT_DIR = RESULTS_DIR / "_realmlp_zlowgal_ckpt"
ID, TARGET = "id", "class"
CLASS_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
GAL, STAR = CLASS_MAP["GALAXY"], CLASS_MAP["STAR"]
SEED, FOLDS = 42, 5
RUN = "realmlp_zlowgal"

# One-sided boost hyperparams
BOOST = 2.5        # loss-weight on low-z GALAXY rows (everything else = 1.0)
STAR_Q = 0.95      # z_thr = this quantile of train-fold STAR redshift

# AMP + fused only (NO compile → real expm4t dropout, clean A/B vs realmlp_deotte)
SPEED_CFG = {"allow_amp": True, "fused_optimizer": True}


def compute_lowz_gal_weights(redshift_tr: np.ndarray, y_tr: np.ndarray,
                             boost: float = BOOST, star_q: float = STAR_Q):
    """One-sided weight: 1.0 everywhere, `boost` on GALAXY rows below the STAR redshift zone.

    z_thr comes from the train-fold STAR redshift only (leak-safe). Strictly ≥1 (no row is
    down-weighted) so GALAXY's bulk and the across-class balance owned by loss_prior_power are
    left intact except for the targeted extra mass on the GAL/STAR confusion rows.
    """
    redshift_tr = np.asarray(redshift_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr)
    star_z = redshift_tr[y_tr == STAR]
    z_thr = float(np.quantile(star_z, star_q)) if len(star_z) else 0.0
    w = np.ones(len(y_tr), dtype=np.float32)
    gal_lowz = (y_tr == GAL) & (redshift_tr < z_thr)
    w[gal_lowz] = boost
    gal_frac = float(gal_lowz.sum()) / max(int((y_tr == GAL).sum()), 1)
    return w, z_thr, gal_frac


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID).reset_index(drop=True)
    redshift_all = train["redshift"].to_numpy(np.float64)  # raw, pre-engineering, id-sorted
    train[TARGET] = train[TARGET].map(CLASS_MAP)
    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID], axis=1)
    test_id = test[ID].to_numpy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    category_map: dict = {}
    X, new_cat_cols, _, _ = feature_engineering(X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, new_num_cols, combo_names = feature_engineering(X_test, cat_cols, num_cols, category_map, fit=False)
    cat_cols = sorted(cat_cols + new_cat_cols)
    num_cols += new_num_cols
    X = X.reindex(sorted(X.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)
    print(f"X {X.shape}  cat {len(cat_cols)}  num {len(num_cols)}  device {CONFIG['device']}")
    print(f"speed: {SPEED_CFG}  |  BOOST {BOOST}  STAR_Q {STAR_Q}")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    n_classes = y.nunique()
    oof = np.zeros((len(X), n_classes), dtype="float32")
    test_proba = np.zeros((len(X_test), n_classes), dtype="float32")
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        fold_seed = SEED + fold * 100
        cfg = {**CONFIG, **SPEED_CFG, "random_state": fold_seed}
        X_tr, X_val, X_tst = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        y_tr, y_val = y.iloc[tri], y.iloc[vai]

        psw, z_thr, gal_frac = compute_lowz_gal_weights(redshift_all[tri], y_tr.to_numpy())
        if fold == 1:
            print(f"  z_thr {z_thr:.4f}  boosted GALAXY frac {gal_frac:.3f}  "
                  f"(rows@{BOOST}x: {int((psw > 1).sum())})")

        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
        tr_enc = enc.fit_transform(X_tr[combo_names], y_tr)
        val_enc = enc.transform(X_val[combo_names])
        tst_enc = enc.transform(X_tst[combo_names])
        te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(n_classes)]
        X_tr[te_names] = tr_enc.astype("float32")
        X_val[te_names] = val_enc.astype("float32")
        X_tst[te_names] = tst_enc.astype("float32")
        X_tr = X_tr.reindex(sorted(X_tr.columns), axis=1)
        X_val = X_val.reindex(sorted(X_val.columns), axis=1)
        X_tst = X_tst.reindex(sorted(X_tst.columns), axis=1)
        if fold == 1:
            print(f"  n_features {X_tr.shape[1]}  cat {len(cat_cols)}  TE {len(te_names)}")

        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        model = RealMLP_TD_Classifier(**cfg)
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols,
                  ckpt_path=str(CKPT_DIR / f"fold{fold}.pth"), X_test=X_tst,
                  per_sample_weight=psw)
        oof[vai] = model.best_val_probs_.astype("float32")
        test_proba += model.predict_proba(X_tst).astype("float32") / FOLDS
        fold_scores.append(float(balanced_accuracy_score(y_val, np.argmax(oof[vai], axis=1))))
        rec_f = recall_score(y_val, np.argmax(oof[vai], axis=1), average=None, labels=[0, 1, 2])
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  recall(G,Q,S) {rec_f.round(4)}")
        del model, X_tr, X_val, X_tst, y_tr, y_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_np = y.to_numpy()
    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y_np, pred))
    rec = recall_score(y_np, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [realmlp_deotte 0.96888; zbalw 0.95923]")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}  "
          f"[want GALAXY↑ vs zbalw 0.9457, STAR floor ~0.93]")
    tw, best = optimize_thresholds(oof, y_np)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, ["GALAXY", "QSO", "STAR"], RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INV_CLASS_MAP[i] for i in np.argmax(test_proba, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_id, TARGET, labels, id_col=ID)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
