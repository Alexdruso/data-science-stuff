"""PR-37: boosted members of a different inductive bias, on the exported fold-safe matrices.

    python src/kind_leg.py --kind mono --folds 1      # results/_pt_<kind>/f{k}_{val,test}.npy

Kinds (none is in the public pool):
  mono    LGBM-xt with monotone constraints (sign of the training-fold Spearman with y, for the
          ratings / teacher / TE / aggregate columns with |rho| > 0.02) — a shape prior that
          label noise cannot bend.
  dart    LGBM DART (dropout boosting), fixed rounds.
  xrank   XGBoost rank:pairwise (pairwise logistic = AUC surrogate), GPU.
  linear  LGBM linear_tree (a ridge model in every leaf: piecewise-linear, not piecewise-constant).
Early stopping (mono, xrank) uses a 5% split of the fold's training rows; the scored fold
selects nothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).resolve().parents[1] / "results"
MONO_PREFIX = ("inflight", "departure_arrival", "ease_", "gate_", "food_", "online_", "seat_",
               "on_board", "leg_room", "baggage", "checkin", "cleanliness", "rating_", "n_high",
               "og_", "TE_", "orig_fd_rate", "aux_ev_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["mono", "dart", "xrank", "linear"], required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[1])
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--rounds", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--mcw", type=float, default=20)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    out = RESULTS / f"_pt_{a.kind}{a.tag}"
    out.mkdir(exist_ok=True)
    for k in a.folds:
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        cols = [str(c) for c in z["cols"]]
        xtr, ytr = z["x_tr"].astype(np.float32), z["y_tr"].astype(int)
        rng = np.random.default_rng(k)
        es = rng.random(len(ytr)) < 0.05
        t0 = time.time()
        if a.kind in ("mono", "dart", "linear"):
            prm = dict(objective="binary", learning_rate=0.02, num_leaves=144, min_child_samples=176,
                       reg_lambda=2.07, colsample_bytree=0.69, subsample=0.87, subsample_freq=1,
                       extra_trees=True, n_jobs=a.jobs, verbose=-1, seed=42 + k)
            if a.kind == "mono":
                mc = []
                for j, c in enumerate(cols):
                    if c.startswith(MONO_PREFIX):
                        ok = ~np.isnan(xtr[::7, j])
                        r = spearmanr(xtr[::7, j][ok], ytr[::7][ok]).statistic
                        mc.append(int(np.sign(r)) if abs(r) > 0.02 else 0)
                    else:
                        mc.append(0)
                print("constrained:", sum(m != 0 for m in mc), flush=True)
                prm.update(monotone_constraints=mc, monotone_constraints_method="advanced")
                dtr = lgb.Dataset(xtr[~es], ytr[~es])
                dva = lgb.Dataset(xtr[es], ytr[es], reference=dtr)
                m = lgb.train(prm, dtr, 6000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                print("best_iter", m.best_iteration, flush=True)
            elif a.kind == "linear":
                prm.update(linear_tree=True, linear_lambda=1.0, num_leaves=31, extra_trees=False,
                           learning_rate=0.03)
                dtr = lgb.Dataset(np.nan_to_num(xtr[~es], nan=-1), ytr[~es])
                dva = lgb.Dataset(np.nan_to_num(xtr[es], nan=-1), ytr[es], reference=dtr)
                m = lgb.train(prm, dtr, 6000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                print("best_iter", m.best_iteration, flush=True)
                z["x_va"], z["x_te"] = np.nan_to_num(z["x_va"], nan=-1), np.nan_to_num(z["x_te"], nan=-1)
            else:
                prm.update(boosting="dart", learning_rate=0.05, drop_rate=0.1, skip_drop=0.5,
                           extra_trees=False)
                m = lgb.train(prm, lgb.Dataset(xtr, ytr), a.rounds)
            pv, pt = m.predict(z["x_va"]), m.predict(z["x_te"])
        else:
            prm = dict(objective="rank:pairwise", eval_metric="auc", learning_rate=a.lr, max_depth=a.depth,
                       min_child_weight=a.mcw, subsample=0.8, colsample_bytree=0.6, reg_lambda=5,
                       tree_method="hist", device="cuda", max_bin=256, seed=42 + k,
                       lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=a.pairs)
            dtr = xgb.DMatrix(xtr[~es], ytr[~es], qid=np.zeros((~es).sum(), dtype=int))
            dva = xgb.DMatrix(xtr[es], ytr[es], qid=np.zeros(es.sum(), dtype=int))
            m = xgb.train(prm, dtr, 8000, evals=[(dva, "es")], early_stopping_rounds=300,
                          verbose_eval=500)
            print("best_iter", m.best_iteration, flush=True)
            r = (0, m.best_iteration + 1)
            pv = m.predict(xgb.DMatrix(z["x_va"]), iteration_range=r)
            pt = m.predict(xgb.DMatrix(z["x_te"]), iteration_range=r)
        print(f"{a.kind} fold {k}: AUC {roc_auc_score(z['y_va'], pv):.5f} ({time.time() - t0:.0f}s)",
              flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), pv]))
        np.save(out / f"f{k}_test.npy", pt)
        del m


if __name__ == "__main__":
    main()
