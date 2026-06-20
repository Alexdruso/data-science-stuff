"""RealMLP + class×redshift-region balanced loss weighting (S6E6 diversity lever).

Extends the single biggest proven lever in this competition — logit-adjust (loss_prior_power),
which took realmlp_deotte's STAR recall 0.909→0.969. That knob balances ACROSS classes. This
adds an orthogonal WITHIN-class balancer: a per-sample loss weight = inverse (class, z-bin) cell
frequency, normalized to mean 1 *within each class*. Effect: the low-z GALAXY bottleneck (GALAXY
mean z≈0.51, so its low-z rows are the rare within-class tail) gets more loss mass, while
loss_prior_power keeps owning the across-class balance untouched (CONFIG class_weight_power=0 →
class_weights is all-ones, so we are NOT double-correcting STAR).

FOLD-SAFE: z-bin edges and cell frequencies are computed on the TRAIN fold only; val/test never
see the weight (it only scales the training loss). No nested CV needed.

Speed (RTX 2060 / Turing): fp16 autocast (tensor cores; softmax+log auto-promote to fp32),
fused AdamW, and torch.compile on the forward. Dropout is held CONSTANT for the compiled run —
the stock expm4t dropout schedule mutates Dropout.p every batch, which would force an Inductor
recompile per step. Everything else matches realmlp_deotte exactly so eval_stack_delta isolates
the weighting's contribution.

GATE: standalone OOF should land near realmlp_deotte's 0.96888 (far off ⇒ alignment/weight bug),
STAR recall must not fall below ~0.93 (the within-class reweight slightly down-weights the
common low-z STAR region — watch the trade). Then eval_stack_delta is the binding decision.

Run:  python src/train_realmlp_zbalw.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from realmlp_deotte import CONFIG, RealMLP_TD_Classifier
from train_realmlp_deotte import feature_engineering, important_combos

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT_DIR = RESULTS_DIR / "_realmlp_zbalw_ckpt"
ID, TARGET = "id", "class"
CLASS_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
SEED, FOLDS = 42, 5
RUN = "realmlp_zbalw"

# z-bin weighting hyperparams
N_ZBINS = 10
WEIGHT_CLIP = (0.2, 5.0)

# Speed knobs layered on top of the stock realmlp_deotte CONFIG. p_drop_sched="constant" is
# required by compile_model (see module docstring); the others are numerically ~neutral.
SPEED_CFG = {
    "allow_amp": True,
    "fused_optimizer": True,
    "compile_model": True,
    "p_drop_sched": "constant",
}


def compute_zbin_weights(redshift_tr: np.ndarray, y_tr: np.ndarray,
                         n_bins: int = N_ZBINS, clip=WEIGHT_CLIP) -> np.ndarray:
    """Per-sample weight = inverse (class, z-bin) cell frequency, mean-1 normalized WITHIN class.

    Quantile bin edges come from the training-fold redshift only (leak-safe). Within each class
    the weights are renormalized to mean 1 so the across-class mass is left to loss_prior_power.
    """
    redshift_tr = np.asarray(redshift_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr)
    edges = np.unique(np.quantile(redshift_tr, np.linspace(0.0, 1.0, n_bins + 1)))
    zbin = np.clip(np.searchsorted(edges, redshift_tr, side="right") - 1, 0, len(edges) - 2)
    w = np.ones(len(y_tr), dtype=np.float64)
    n_zb = len(edges) - 1
    for c in np.unique(y_tr):
        mask = y_tr == c
        bins_c = zbin[mask]
        counts = np.bincount(bins_c, minlength=n_zb).astype(np.float64)
        inv = 1.0 / np.clip(counts[bins_c], 1.0, None)
        inv = inv / inv.mean()                    # mean-1 within class
        inv = np.clip(inv, clip[0], clip[1])      # bound extremes (clip is approximate)
        w[mask] = inv / inv.mean()                # restore strict mean-1 (the binding invariant)
    return w.astype(np.float32)


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
    print(f"speed: {SPEED_CFG}  |  z-bins {N_ZBINS}  clip {WEIGHT_CLIP}")

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

        psw = compute_zbin_weights(redshift_all[tri], y_tr.to_numpy())
        if fold == 1:
            print(f"  z-bin weight: min {psw.min():.3f} max {psw.max():.3f} mean {psw.mean():.3f}")

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
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [realmlp_deotte 0.96888]")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}  [STAR floor ~0.93]")
    tw, best = optimize_thresholds(oof, y_np)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, ["GALAXY", "QSO", "STAR"], RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INV_CLASS_MAP[i] for i in np.argmax(test_proba, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID: test_id, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
