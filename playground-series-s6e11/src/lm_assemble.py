"""Assemble an LM leg from cached per-fold logits on the CPU (no GPU / tokenizer needed).

    python src/lm_assemble.py lm_distilgpt2_ep10
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def main() -> None:
    name = sys.argv[1]
    out = RESULTS_DIR / f"_{name}"
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    y = tr[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    oof, te = np.zeros(len(y)), np.zeros(len(te_ids))
    for k in range(1, 6):
        oof[folds == k] = np.load(out / f"f{k}_val.npy")
        te += np.load(out / f"f{k}_test.npy") / 5
        print(f"  fold {k}: {roc_auc_score(y[folds == k], oof[folds == k]):.5f}")
    auc = float(roc_auc_score(y, oof))
    print(f"{name}: OOF AUC {auc:.5f}")
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    save_predictions(RESULTS_DIR, name, oof=sig(oof), test=sig(te), train_ids=tr["id"].to_numpy(),
                     test_ids=te_ids, folds=folds)
    save_cv_result(RESULTS_DIR, name, [], auc, metric_name="auc", params={"assembled": True})
    write_submission(SUBMISSIONS_DIR, f"{name}.csv", te_ids, TARGET, sig(te))


if __name__ == "__main__":
    main()
