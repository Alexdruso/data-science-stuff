"""PR-16 read: do top-k truncation features add to [pool_v7 logit, LM LLR] on the fold?

    python src/topk_eval.py _lm_distilgpt2_ep10 --fold 1 [--base pool_v7]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--base", default="pool_v7")
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    z = np.load(RESULTS_DIR / f"{a.dir}_topk" / f"f{a.fold}_val_feats.npz")
    rows, F, names, llr = z["rows"], z["F"], list(z["names"]), z["llr"]
    ys = y[rows]
    p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.base, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = logit(p0)[rows]
    keep = F.std(0) > 0
    F, names = F[:, keep], [n for n, k in zip(names, keep) if k]
    Fs = StandardScaler().fit_transform(np.log1p(np.abs(F)) * np.sign(F))

    def stack(x: np.ndarray, C: float = 1.0) -> float:
        accs = []
        for seed in (0, 1, 2):
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            p = cross_val_predict(LogisticRegression(C=C, max_iter=5000), x, ys, cv=cv,
                                  method="decision_function")
            accs.append(roc_auc_score(ys, p))
        return float(np.mean(accs))

    b2 = np.column_stack([base, llr])
    r0 = stack(base[:, None])
    r1 = stack(b2)
    print(f"base {r0:.6f}  base+llr {r1:.6f} ({r1 - r0:+.6f})  llr solo {roc_auc_score(ys, llr):.5f}")
    for C in (0.01, 0.1, 1.0):
        r = stack(np.column_stack([b2, Fs]), C)
        print(f"  + all {len(names)} topk feats C {C}: {r:.6f}  gain {r - r1:+.6f}")
    groups = {g: [i for i, n in enumerate(names) if n.startswith(g)]
              for g in ("nout10", "nout20", "nout50", "nout100", "maxlrank", "excl")}
    for g, idx in groups.items():
        if idx:
            r = stack(np.column_stack([b2, Fs[:, idx]]))
            print(f"  + {g:9s} ({len(idx)}): gain {r - r1:+.6f}")


if __name__ == "__main__":
    main()
