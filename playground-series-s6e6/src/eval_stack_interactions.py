"""Non-destructive test: redshift-region-gated flow in the LR meta (S6E6).

The flow base is decorrelated (STAR recall 0.869 where class densities overlap) but too WEAK
globally (0.93) → a GLOBAL LR meta down-weights it to ~0 and the +flow stack-delta was flat.
Hypothesis (06-14 meta brainstorm idea #1): let the meta weight the flow ONLY in the low-z
STAR/GALAXY-overlap region, by feeding flow_logit_c × I(z in bin_b) instead of a plain global
flow term. The meta then learns a per-region flow weight (near-zero where the flow is bad,
nonzero in low-z where it's right).

Mirrors eval_stack_delta's meta EXACTLY (LR C=1.0, class_weight=balanced, 5 seeds × 5-fold,
threshold-tuned readout) so the numbers are directly comparable. Prints several configs in one
run. PRINTS ONLY — writes nothing.

Run:  python src/eval_stack_interactions.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from postprocess import optimize_thresholds

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEEDS = [2024, 7, 13, 42, 99]
EPS = 1e-6
BASE_MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe", "xgb_deotte", "realmlp_deotte",
               "catboost_deotte", "catboost_v3", "xgb_v3fe", "chain_cascade", "chain_cascade_xgb"]
FLOW = "flow"
STAR_ZONE_THR = 0.178   # STAR's ~95th-pct redshift = the GAL/STAR confusion zone (from zlowgal)


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def load_logits(name: str) -> np.ndarray:
    return logit(np.load(RESULTS_DIR / f"oof_{name}.npy"))


def region_indicators(z: np.ndarray, scheme: str) -> tuple[np.ndarray, list[str]]:
    """Return one-hot region indicators (n, n_regions) and labels for a binning scheme."""
    if scheme == "starzone":
        ind = np.stack([(z < STAR_ZONE_THR), (z >= STAR_ZONE_THR)], axis=1).astype(np.float64)
        return ind, [f"z<{STAR_ZONE_THR}", f"z>={STAR_ZONE_THR}"]
    nb = int(scheme[4:])  # "qbinK"
    edges = np.unique(np.quantile(z, np.linspace(0, 1, nb + 1)))
    code = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    ind = np.zeros((len(z), nb), dtype=np.float64)
    ind[np.arange(len(z)), code] = 1.0
    return ind, [f"qbin{nb}_{b}" for b in range(nb)]


def eval_meta(Z: np.ndarray, y: np.ndarray, label: str) -> float:
    oof = np.zeros((len(y), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Z, y):
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            lr.fit(Z[tri], y[tri])
            oof[vai] += lr.predict_proba(Z[vai]) / len(SEEDS)
    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    _, best = optimize_thresholds(oof, y)
    print(f"  {label:<34} cols {Z.shape[1]:>3}  argmax {argmax:.6f}  thresh {best:.6f}  "
          f"G/Q/S {rec[0]:.4f}/{rec[1]:.4f}/{rec[2]:.4f}")
    return best


def main() -> None:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = LabelEncoder().fit_transform(tr[TARGET].to_numpy())
    z = tr["redshift"].to_numpy().astype(np.float64)

    Z_base = np.hstack([load_logits(m) for m in BASE_MODELS])   # (n, 33)
    Z_flow = load_logits(FLOW)                                  # (n, 3)
    print(f"base {Z_base.shape}  flow {Z_flow.shape}  n {len(y)}")

    print("\n── references ──")
    base_thr = eval_meta(Z_base, y, "baseline-11")
    flow_thr = eval_meta(np.hstack([Z_base, Z_flow]), y, "+flow (plain global)")

    print("\n── region-gated flow (flow_logit × z-region indicator) ──")
    best_label, best_val = None, -1.0
    for scheme in ["starzone", "qbin3", "qbin4", "qbin6"]:
        ind, names = region_indicators(z, scheme)
        # flow contributes ONLY through region-gated terms (no plain flow term → no redundancy)
        inter = np.hstack([Z_flow * ind[:, b: b + 1] for b in range(ind.shape[1])])
        thr = eval_meta(np.hstack([Z_base, inter]), y, f"+flow×{scheme}")
        if thr > best_val:
            best_label, best_val = scheme, thr

    print(f"\nbaseline-11 thresh {base_thr:.6f}  |  +flow {flow_thr:.6f}  |  "
          f"best gated (+flow×{best_label}) {best_val:.6f}")
    print(f"Δ best-gated vs baseline-11: {best_val - base_thr:+.6f}  "
          f"(noise floor ~0.0014; need a clear, consistent beat)")


if __name__ == "__main__":
    main()
