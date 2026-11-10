"""Fold-level read of LM fold outputs against a saved pooled stack (single-logit base).

    python src/lm_pool_eval.py pool_noLM _lm_distilgpt2_ep10 _lm_distilgpt2_allneg [--fold 1]

Prints each LM's solo AUC on the fold and the inner-CV gain of LR([base logit, LM]) over the
base alone; also the gain of each LM on top of masayakawamata's public GPT-2 + base.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    m = fold_assignment(y) == a.fold
    ys = y[m]
    p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.base, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = logit(p0)[m][:, None]
    masa = np.load("ext/dl/s6e11-gpt2-target-first-oof/oof.npy")[m][:, None]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def stack(x: np.ndarray) -> float:
        p = cross_val_predict(LogisticRegression(max_iter=3000), x, ys, cv=cv,
                              method="decision_function")
        return float(roc_auc_score(ys, p))

    b0, bm = stack(base), stack(np.hstack([base, masa]))
    print(f"fold {a.fold}: base {b0:.6f}  base+masa {bm:.6f}  masa solo "
          f"{roc_auc_score(ys, masa[:, 0]):.5f}")
    for d in a.dirs:
        v = np.load(RESULTS_DIR / d / f"f{a.fold}_val.npy")[:, None]
        if v.min() >= 0 and v.max() <= 1:  # probabilities -> log-odds
            v = np.clip(v, 1e-9, 1 - 1e-9)
            v = logit(v)
        print(f"  {d:40s} solo {roc_auc_score(ys, v[:, 0]):.5f}  gain {stack(np.hstack([base, v])) - b0:+.6f}"
              f"  gain over base+masa {stack(np.hstack([base, masa, v])) - bm:+.6f}"
              f"  rho(masa) {np.corrcoef(v[:, 0], masa[:, 0])[0, 1]:.4f}")


if __name__ == "__main__":
    main()
