"""EBM / GA2M leg (interpret): additive shape functions + pairwise interactions —
matches the d2-trees-are-best finding (generator looks additive + pairwise).

    python src/ebm_leg.py [--folds 1 2 3 4 5] [--inter 30] [--jobs 3]
Per-fold outputs cached in results/_ebm_i<inter>/f{k}_{val,test}.npy; assembled when all 5 exist.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]


def digit_keys(path: Path) -> "pd.DataFrame":
    """Label-free digit keys of each float as written in the CSV (the generator's tokens)."""
    import pandas as pd

    raw = pd.read_csv(path, usecols=["id", *FLOATS], dtype=str)
    out = pd.DataFrame({"id": raw["id"].astype(np.int64)})
    for c in FLOATS:
        s = raw[c].fillna("nan")
        ip = s.str.split(".").str[0]
        fr = s.str.split(".").str[1].fillna("")
        out[f"{c}_d1"] = fr.str[:1].replace("", "x")
        out[f"{c}_d2"] = fr.str[1:2].replace("", "x")
        out[f"{c}_nd"] = fr.str.len().astype(str)
        out[f"{c}_il"] = ip.str.len().astype(str)
        out[f"{c}_lead"] = ip.str[:1]
        out[f"{c}_last"] = ip.str[-1:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--inter", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mb", type=int, default=1024, help="max_bins")
    ap.add_argument("--mib", type=int, default=64, help="max_interaction_bins")
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--leaves", type=int, default=2)
    ap.add_argument("--smooth", type=int, default=75)
    ap.add_argument("--tag", default="")
    ap.add_argument("--digits", action="store_true", help="add raw-string digit keys of the floats")
    a = ap.parse_args()
    name = f"ebm_i{a.inter}" + (f"_s{a.seed}" if a.seed != 42 else "") + ("_dig" if a.digits else "") + a.tag
    out = RESULTS_DIR / f"_{name}"
    out.mkdir(exist_ok=True)
    train = build_features(pl.read_csv(DATA_DIR / "train.csv")).to_pandas()
    test = build_features(pl.read_csv(DATA_DIR / "test.csv")).to_pandas()
    if a.digits:
        train = train.merge(digit_keys(DATA_DIR / "train.csv"), on="id", how="left")
        test = test.merge(digit_keys(DATA_DIR / "test.csv"), on="id", how="left")
    feats = [c for c in train.columns if c not in EXCLUDE_COLS]
    y = train[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    for k in a.folds:
        if (out / f"f{k}_val.npy").exists():
            continue
        t0 = time.time()
        tr, va = folds != k, folds == k
        m = ExplainableBoostingClassifier(interactions=a.inter, outer_bags=a.bags, n_jobs=a.jobs,
                                          random_state=a.seed + k, max_bins=a.mb,
                                          max_interaction_bins=a.mib, learning_rate=a.lr,
                                          max_leaves=a.leaves, smoothing_rounds=a.smooth)
        m.fit(train.loc[tr, feats], y[tr])
        pv = m.predict_proba(train.loc[va, feats])[:, 1]
        print(f"  fold {k}: auc {roc_auc_score(y[va], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out / f"f{k}_val.npy", pv)
        np.save(out / f"f{k}_test.npy", m.predict_proba(test[feats])[:, 1])
    if all((out / f"f{k}_val.npy").exists() for k in range(1, 6)):
        oof = np.zeros(len(y))
        te = np.zeros(len(test))
        for k in range(1, 6):
            oof[folds == k] = np.load(out / f"f{k}_val.npy")
            te += np.load(out / f"f{k}_test.npy") / 5
        auc = float(roc_auc_score(y, oof))
        print(f"{name}: OOF AUC {auc:.5f}")
        save_predictions(RESULTS_DIR, name, oof=oof, test=te, train_ids=train["id"].to_numpy(),
                         test_ids=test["id"].to_numpy(), folds=folds)
        save_cv_result(RESULTS_DIR, name, [], auc, metric_name="auc", params=vars(a))
        write_submission(SUBMISSIONS_DIR, f"{name}.csv", test["id"].to_numpy(), TARGET, te)


if __name__ == "__main__":
    main()
