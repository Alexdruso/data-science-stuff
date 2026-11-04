"""Assemble replica chunk scores: fold-level read now, full leg (oof/test parquet) when complete.

    python src/replica_assemble.py great_e80 [--base pool_v4]
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
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions, save_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def load(d: Path, split: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    s, have = np.full(n, np.nan), np.zeros(n, bool)
    for f in sorted(d.glob(f"{split}_*.npz")):
        z = np.load(f)
        lse = np.logaddexp.reduce(z["lp"].astype(float), axis=1)
        s[z["rows"]] = lse[1] - lse[0]
        have[z["rows"]] = True
    return s, have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--base", default="pool_v4")
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    d = RESULTS_DIR / "_replica" / f"{a.ckpt}_score"
    s, have = load(d, "tr", len(y))
    p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.base, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = np.log(p0 / (1 - p0))
    print(f"scored train rows: {have.sum()}; solo AUC {roc_auc_score(y[have], s[have]):.5f}")
    for k in range(1, 6):
        m = have & (folds == k)
        if m.sum() < (folds == k).sum():
            continue
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        st = lambda x: roc_auc_score(y[m], cross_val_predict(LogisticRegression(max_iter=3000), x, y[m], cv=cv, method="decision_function"))  # noqa: E731
        b0, b1 = st(base[m][:, None]), st(np.column_stack([base[m], s[m]]))
        c = LogisticRegression(max_iter=3000).fit(np.column_stack([base[m], s[m]]), y[m]).coef_[0]
        print(f"fold {k}: base {b0:.6f}  + replica {b1:.6f}  gain {b1 - b0:+.6f}  coefs {np.round(c, 3)}")
    st_, hte = load(d, "te", len(te_ids))
    if have.all() and hte.all():
        prior = np.log(y.mean() / (1 - y.mean()))
        sig = lambda v: 1 / (1 + np.exp(-(v + prior)))  # noqa: E731
        name = f"replica_{a.ckpt}"
        save_predictions(RESULTS_DIR, name, oof=sig(s), test=sig(st_), train_ids=ids, test_ids=te_ids)
        save_cv_result(RESULTS_DIR, name, [], float(roc_auc_score(y, s)), metric_name="auc",
                       params={"ckpt": a.ckpt, "note": "orig-only replica, no train labels"})
        print(f"saved leg {name}")


if __name__ == "__main__":
    main()
