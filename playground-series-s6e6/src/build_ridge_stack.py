"""Ridge meta-learner over base OOF logits — the 'ridge' half of the winning 0.96973 recipe.

The reference `single-realmlp-0-96973-ridge-pl` calibrates with RIDGE, not logistic regression.
Ridge differs from our LR stacker (build_lr_stack, 0.96977) in two ways that matter for balanced
accuracy at argmax: (1) it minimizes SQUARED loss to the one-hot target (a Brier-type objective
directly aligned with argmax) rather than cross-entropy, fit per-class independently; (2) it's L2
least-squares so it's very stable on the OOF. We add inverse-frequency SAMPLE WEIGHTS as the
balanced-accuracy analog of class_weight="balanced" (worth +0.008 historically).

Honest 5-fold × multi-seed OOF stacking (fit on 4 folds of the OOF rows, predict the 5th), same as
build_lr_stack. Sweeps alpha so we read the whole curve, not a cherry-picked point. Inputs default
to base logits; --probs uses raw probabilities; --rawfeats appends redshift+colors (region info).

GATE: OOF argmax must beat LR-stack 0.96977 / GBDT-stack 0.96981 to be a real lever.

Run:  python src/build_ridge_stack.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte"]
SEEDS = [2024, 7, 13, 42, 99]
ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def inv_freq_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    w = len(y) / (3.0 * np.maximum(counts, 1.0))
    return w[y]


def run_ridge(Ztr, Zte, y, alpha, seeds):
    """Honest OOF ridge meta at one alpha. Returns (oof_probs, test_probs)."""
    oof = np.zeros((len(y), 3))
    test = np.zeros((Zte.shape[0], 3))
    Y = np.eye(3)[y]
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Ztr, y):
            sc = StandardScaler().fit(Ztr[tri])
            Xtr, Xva, Xte = sc.transform(Ztr[tri]), sc.transform(Ztr[vai]), sc.transform(Zte)
            r = Ridge(alpha=alpha)
            r.fit(Xtr, Y[tri], sample_weight=inv_freq_weights(y[tri]))
            oof[vai] += softmax(r.predict(Xva)) / len(seeds)
            test += softmax(r.predict(Xte)) / (len(seeds) * 5)
    return oof, test


def main() -> None:
    use_probs = "--probs" in sys.argv
    rawfeats = "--rawfeats" in sys.argv
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    Ztr, Zte, used = [], [], []
    for m in MODELS:
        fo, ft = RESULTS_DIR / f"oof_{m}.npy", RESULTS_DIR / f"test_{m}.npy"
        if fo.exists() and ft.exists():
            o, t = np.load(fo), np.load(ft)
            Ztr.append(o if use_probs else logit(o))
            Zte.append(t if use_probs else logit(t))
            used.append(m)
    Ztr, Zte = np.hstack(Ztr), np.hstack(Zte)
    if rawfeats:
        cols = ["redshift", "u_g", "g_r", "r_i", "i_z", "u_z"]
        Ztr = np.hstack([Ztr, tr.select(cols).to_numpy()])
        Zte = np.hstack([Zte, te.select(cols).to_numpy()])
    kind = "probs" if use_probs else "logits"
    kind += "+raw" if rawfeats else ""
    print(f"Ridge stack over {len(used)} models ({kind}, dim {Ztr.shape[1]}): {used}")
    print(f"  [targets: LR-stack 0.96977 / GBDT-stack 0.96981]")

    best_overall = (-1.0, None, None, None)
    for alpha in ALPHAS:
        oof, test = run_ridge(Ztr, Zte, y, alpha, SEEDS)
        am = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
        _, thr = optimize_thresholds(oof, y)
        mark = " <" if am > 0.96977 else ""
        print(f"  alpha {alpha:6.1f}: argmax {am:.5f}  thresh {thr:.5f}{mark}")
        if am > best_overall[0]:
            best_overall = (am, alpha, oof, test)

    am, alpha, oof, test = best_overall
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"\nBEST alpha {alpha}: argmax {am:.5f}  thresh {best:.5f}")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}")

    if am > 0.96977:
        save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / "threshold_weights_ridgestack.json")
        save_cv_result(RESULTS_DIR, "ridgestack", [], best, metric_name="balanced_acc")
        np.save(RESULTS_DIR / "oof_ridgestack.npy", oof)
        np.save(RESULTS_DIR / "test_ridgestack.npy", test)
        labels = le.inverse_transform(np.argmax(test * tw, axis=1))
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / "ridgestack.csv", index=False)
        print("Saved → ridgestack (beat LR stack)")
    else:
        print("Did not beat LR stack — not saved as submission.")


if __name__ == "__main__":
    main()
