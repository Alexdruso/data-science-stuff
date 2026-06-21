"""Model-free diagnostic: does small-k retrieval against the REAL SDSS catalog carry
incremental signal beyond the (qbin-TE-saturated) stack? (S6E6)

Logic. The stack already contains models built on Deotte's COARSE qbin target-encoding of
colors/redshift. `knnorig` tried the real catalog but at k=200 — that smoothing collapses to
≈ the qbin TE. The untested degree of freedom is RESOLUTION: at small k (5-25) a euclidean
retrieval in standardized colour+redshift space resolves the JOINT local boundary the marginal
qbins blur — exactly where the low-z STAR/GALAXY confusion lives.

This script does NOT train anything. It builds a NearestNeighbors index on the real catalog,
queries every (id-sorted) train row once at k=200, then evaluates per-class local PURITY at
k∈{5,10,25,200}. The load-bearing test (advisor-prescribed, mirrors the lineage oracle / low-z
AUC gates): on the rows the lrstack gets WRONG, does small-k real-neighbour purity point to the
TRUE label above the always-majority baseline — and does small k beat large k (the resolution
claim)? Both raw-purity-argmax and real-prior-corrected (enrichment) argmax are reported.

Run:  python src/diag_knn_retrieval.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from deotte_features import CLASS_TO_INT, CLASSES

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
KS = [5, 10, 25, 200]


def color_space(df: pd.DataFrame) -> np.ndarray:
    """Standardisable colour+redshift view: 4 SDSS colours + redshift (brightness nuisance removed)."""
    u, g, r, i, z = (df[c].to_numpy(np.float64) for c in ("u", "g", "r", "i", "z"))
    return np.column_stack([u - g, g - r, r - i, i - z, df["redshift"].to_numpy(np.float64)])


def purity_argmax(counts: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """argmax of per-class neighbour fraction; if prior given, divide by it first (enrichment)."""
    frac = counts / counts.sum(axis=1, keepdims=True)
    if prior is not None:
        frac = frac / prior
    return np.argmax(frac, axis=1)


def main() -> None:
    # --- align everything to id-sorted order (oof_lrstack convention) ---
    tr = pd.read_csv(DATA_DIR / "train.csv").sort_values("id").reset_index(drop=True)
    y = tr["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y_ref = orig["class"].str.upper().map(CLASS_TO_INT).astype("int64").to_numpy()
    ref_prior = np.array([(y_ref == c).mean() for c in range(3)])
    print(f"train {len(tr)}  real-catalog {len(orig)}  ref prior G/Q/S "
          f"{ref_prior[0]:.3f}/{ref_prior[1]:.3f}/{ref_prior[2]:.3f}")

    # --- standardise on the reference space, build index, query train once at max k ---
    scaler = StandardScaler().fit(color_space(orig))
    ref = scaler.transform(color_space(orig)).astype(np.float32)
    qry = scaler.transform(color_space(tr)).astype(np.float32)
    print(f"fitting NearestNeighbors on {ref.shape} ...", flush=True)
    nn = NearestNeighbors(n_neighbors=max(KS), algorithm="auto", n_jobs=-1).fit(ref)
    print("querying train rows (k=200) ...", flush=True)
    _, idx = nn.kneighbors(qry, n_neighbors=max(KS))  # (n_train, 200) sorted by distance
    nbr_lbl = y_ref[idx]  # (n_train, 200) neighbour true labels

    # --- precompute per-k class counts from the sorted neighbour labels ---
    onehot = np.eye(3, dtype=np.int32)[nbr_lbl]  # (n_train, 200, 3)
    cum = np.cumsum(onehot, axis=1)              # counts among first j neighbours

    # --- the stack we are testing against ---
    stack_pred = np.argmax(np.load(RESULTS_DIR / "oof_lrstack.npy"), axis=1)
    wrong = stack_pred != y
    nW = int(wrong.sum())
    # baselines among the wrong rows
    maj_among_wrong = np.bincount(y[wrong], minlength=3)
    base_acc = maj_among_wrong.max() / nW
    print(f"\nlrstack OOF: {(~wrong).mean():.5f} acc; {nW} wrong rows "
          f"(true G/Q/S {maj_among_wrong[0]}/{maj_among_wrong[1]}/{maj_among_wrong[2]}; "
          f"always-majority baseline on wrong = {base_acc:.4f})")

    print("\n         |  ALL TRAIN          |  STACK-WRONG ROWS (the test)")
    print("  k mode | acc     bal_acc     | recover-true-acc   vs base   conf>0.6:n/acc")
    for k in KS:
        counts = cum[:, k - 1, :]  # (n_train, 3) counts among k nearest
        for mode, prior in (("raw ", None), ("enr ", ref_prior)):
            pred = purity_argmax(counts, prior)
            acc_all = (pred == y).mean()
            ba_all = balanced_accuracy_score(y, pred)
            rec = (pred[wrong] == y[wrong]).mean()
            # confidence gate: top-class fraction (raw) > 0.6 among wrong rows
            frac = counts / counts.sum(axis=1, keepdims=True)
            conf = frac.max(axis=1) > 0.6
            cw = conf & wrong
            cw_n = int(cw.sum())
            cw_acc = (pred[cw] == y[cw]).mean() if cw_n else float("nan")
            print(f"{k:3d} {mode}| {acc_all:.4f}  {ba_all:.4f}      | "
                  f"{rec:.4f}            {rec - base_acc:+.4f}    {cw_n:6d}/{cw_acc:.4f}")


if __name__ == "__main__":
    main()
