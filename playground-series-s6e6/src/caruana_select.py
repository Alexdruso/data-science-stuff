"""Caruana greedy ensemble selection over ALL accumulated OOF arrays (S6E6).

The one standard Kaggle combiner we never ran. Greedy forward selection WITH REPLACEMENT
(Caruana et al. 2004) over the library of base OOF probability arrays, averaging probabilities
and optimizing balanced accuracy (argmax). HONEST: selection is done inside an outer 5-fold so
the reported score is held-out, not best-on-OOF (naive best-on-OOF manufactures a fake gain —
our own documented selection-bias trap).

This only RE-WEIGHTS existing base logits (no new information) → ceiling is ~noise. Treat as
tidy-up / close-out of the combiner axis, NOT a lever. Compares against the production LR-stack
baseline (argmax 0.970410). PRINTS ONLY — writes nothing.

Excludes meta/combination arrays (lrstack/gbdtstack/metablend/altmeta/ensemble) — including them
would be circular (greedy would just pick the stack and report ≈stack).

Run:  python -u src/caruana_select.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.stacking import caruana_select

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_OUTER = 5
N_STEPS = 50          # greedy picks (with replacement)
INIT_TOPK = 3         # seed ensemble with the K best singletons (Caruana's bagged-start lite)
SEED = 42

# meta / combination outputs — exclude (circular)
META_BLOCKLIST = {
    "lrstack", "gbdtstack", "gbdtstack_lineage", "metablend",
    "altmeta_nn", "altmeta_ridgecal-gbdt", "ensemble",
}


def load_library() -> tuple[list[str], np.ndarray]:
    names, arrs = [], []
    for fp in sorted(RESULTS_DIR.glob("oof_*.npy")):
        name = fp.stem[len("oof_"):]
        if name in META_BLOCKLIST:
            continue
        a = np.load(fp)
        if a.ndim != 2 or a.shape[1] != 3:
            continue
        arrs.append(a.astype(np.float64))
        names.append(name)
    return names, np.stack(arrs, axis=0)   # (M, N, 3)


def main() -> None:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = LabelEncoder().fit_transform(tr[TARGET].to_numpy())
    names, P = load_library()
    print(f"library: {len(names)} base arrays, N={P.shape[1]} rows")
    print(f"(excluded meta: {sorted(META_BLOCKLIST)})\n")

    # Greedy selection + honest outer-CV gate: the shared implementation in
    # data_science_stuff.kaggle.stacking.caruana_select.
    result = caruana_select(
        P, y, n_steps=N_STEPS, init_topk=INIT_TOPK, n_outer=N_OUTER, seed=SEED
    )
    for fold, sc in enumerate(result.holdout_scores, 1):
        print(f"  outer fold {fold}  held-out bal_acc {sc:.6f}")

    mean_ho = float(np.mean(result.holdout_scores))
    print(f"\nNESTED held-out mean: {mean_ho:.6f} ± {np.std(result.holdout_scores):.6f}")

    # full-data (optimistic) composition for inspection
    weights: dict[str, int] = {}
    for p in result.full_picks:
        weights[names[p]] = weights.get(names[p], 0) + 1
    comp = ", ".join(f"{k}:{v}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
    print(f"\nFULL-data greedy (OPTIMISTIC, selection-biased): {result.full_score:.6f}")
    print(f"  composition (counts/{N_STEPS}): {comp}")

    print(f"\n{'='*60}")
    print(f"LR-stack baseline (production, argmax): 0.970410")
    print(f"Caruana nested held-out (honest):      {mean_ho:.6f}")
    print(f"Δ vs LR-stack: {mean_ho - 0.970410:+.6f}  (prob-averaging < stacking expected; "
          f"noise floor ~0.0014)")


if __name__ == "__main__":
    main()
