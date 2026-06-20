"""KNN soft-probability base — a creative diversity lever (S6E6 campaign).

Instance-based family, orthogonal to GBDT/NN: per-class soft probabilities from the k nearest
neighbours in standardized photometry+redshift space. The 240-feature recipe has ZERO distance/
neighbour features, and axis-aligned GBDT splits can't represent local-manifold density — so even
a moderate KNN base may DECORRELATE (the only thing the stacker rewards). Two label sources, both
leak-safe: (a) train-fold neighbours (fold-safe), (b) the real SDSS17 originals (external).
Class-balanced soft vote (inverse-freq weighting) + distance weighting. id-sorted I/O.

GATE: not standalone strength (expected ~0.94-0.95) but STACK CONTRIBUTION + error-decorrelation
vs the strong bases. Saves oof_knn/test_knn (train-neighbour) and oof_knnorig/test_knnorig (original).

Run:  python src/train_knn_base.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED, N_SPLITS, K = 42, 5, 200
KNN_FEATS = ["u", "g", "r", "i", "z", "redshift"]
EPS = 1e-6


def colors(df: pd.DataFrame) -> np.ndarray:
    base = df[KNN_FEATS].to_numpy(np.float32)
    col = np.stack([df["u"] - df["g"], df["g"] - df["r"], df["r"] - df["i"],
                    df["i"] - df["z"]], axis=1).astype(np.float32)
    return np.hstack([base, col])


def soft_vote(nn: NearestNeighbors, y_ref: np.ndarray, Xq: np.ndarray, cw: np.ndarray) -> np.ndarray:
    """Distance- and class-balance-weighted soft KNN class probabilities for query rows."""
    dist, idx = nn.kneighbors(Xq)
    w = 1.0 / (dist + EPS)  # distance weighting
    out = np.zeros((len(Xq), len(CLASSES)), dtype=np.float32)
    lab = y_ref[idx]
    for c in range(len(CLASSES)):
        out[:, c] = (w * (lab == c) * cw[c]).sum(axis=1)
    return out / out.sum(axis=1, keepdims=True).clip(min=EPS)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()
    cw = len(y) / (len(CLASSES) * np.bincount(y, minlength=len(CLASSES)))

    Xtr_raw, Xte_raw, Xorig_raw = colors(train), colors(test), colors(orig)

    # ── (b) original-data KNN (external → leak-safe, same for all folds) ──────────
    sc_o = StandardScaler().fit(Xorig_raw)
    nn_o = NearestNeighbors(n_neighbors=K).fit(sc_o.transform(Xorig_raw))
    cw_o = len(y_orig) / (len(CLASSES) * np.bincount(y_orig, minlength=len(CLASSES)))
    oof_orig = soft_vote(nn_o, y_orig, sc_o.transform(Xtr_raw), cw_o)
    test_orig = soft_vote(nn_o, y_orig, sc_o.transform(Xte_raw), cw_o)
    print(f"orig-KNN argmax {balanced_accuracy_score(y, oof_orig.argmax(1)):.5f}")

    # ── (a) train-fold KNN (fold-safe) ───────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(Xtr_raw, y), 1):
        sc = StandardScaler().fit(Xtr_raw[tri])
        nn = NearestNeighbors(n_neighbors=K).fit(sc.transform(Xtr_raw[tri]))
        oof[vai] = soft_vote(nn, y[tri], sc.transform(Xtr_raw[vai]), cw)
        test_proba += soft_vote(nn, y[tri], sc.transform(Xte_raw), cw) / N_SPLITS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\ntrain-KNN OOF argmax: {argmax:.5f}  (expected weak ~0.94-0.95; value = DECORRELATION)")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)

    for name, o, t in [("knn", oof, test_proba), ("knnorig", oof_orig, test_orig)]:
        np.save(RESULTS_DIR / f"oof_{name}.npy", o)
        np.save(RESULTS_DIR / f"test_{name}.npy", t)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / "threshold_weights_knn.json")
    save_cv_result(RESULTS_DIR, "knn", fold_scores, best, metric_name="balanced_acc")
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / "knn.csv", index=False)
    print("Saved → knn + knnorig (oof/test). Add to stack to test decorrelation.")


if __name__ == "__main__":
    main()
