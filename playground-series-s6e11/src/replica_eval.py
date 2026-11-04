"""Read replica scores on the fold-1 subset: solo AUC, corr with LMs, gain over pool logit.

    python src/replica_eval.py pool_v4 great_e10_T0.7_k50 [...]
"""

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

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    base_name, names = sys.argv[1], sys.argv[2:]
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    rows = np.load(RESULTS_DIR / "_replica" / "rows_f1.npy")
    va = np.where(fold_assignment(y) == 1)[0]
    pos_in_va = np.searchsorted(va, rows)
    ys = y[rows]
    p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, base_name, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = np.log(p0 / (1 - p0))[rows][:, None]
    masa = np.load("ext/dl/s6e11-gpt2-target-first-oof/oof.npy")[rows]
    ep10 = np.load(RESULTS_DIR / "_lm_distilgpt2_ep10" / "f1_val.npy")[pos_in_va]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def st(x: np.ndarray) -> float:
        p = cross_val_predict(LogisticRegression(max_iter=3000), x, ys, cv=cv, method="decision_function")
        return float(roc_auc_score(ys, p))

    b0 = st(base)
    print(f"{len(rows)} rows; pool {roc_auc_score(ys, base[:, 0]):.5f} (stacked {b0:.5f})  "
          f"masa {roc_auc_score(ys, masa):.5f}  ep10 {roc_auc_score(ys, ep10):.5f}  "
          f"masa gain {st(np.column_stack([base, masa])) - b0:+.6f}")
    for n in names:
        s = np.load(RESULTS_DIR / "_replica" / f"{n}_f1sub.npy")
        print(f"  {n:28s} solo {roc_auc_score(ys, s):.5f}  rho(masa) {np.corrcoef(s, masa)[0, 1]:.3f}  "
              f"gain {st(np.column_stack([base, s])) - b0:+.6f}")


if __name__ == "__main__":
    main()
