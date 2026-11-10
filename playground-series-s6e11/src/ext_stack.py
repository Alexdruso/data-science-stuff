"""Stack our legs with public OOF banks (same folds: StratifiedKFold(5, True, 42), id-sorted rows).

    python src/ext_stack.py [--ext ext/goodpjw ...] [--ours a b c] [--out name] [--C 1]
Members enter as logit(rank); meta = LR, outer StratifiedKFold(5, seed 7) x 3 seeds as stack.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import load_predictions, save_predictions
from data_science_stuff.kaggle.stacking import nested_lr_stack, rank_logit

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def load_ext(dirs: list[str], n_tr: int, n_te: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*_oof.npy")):
            base = p.name[: -len("_oof.npy")]
            t = next((p.parent / f"{base}_{s}.npy" for s in ("full", "test")
                      if (p.parent / f"{base}_{s}.npy").exists()), None)
            if t is None:
                continue
            o, te = np.load(p).ravel(), np.load(t).ravel()
            if len(o) == n_tr and len(te) == n_te and np.isfinite(o).all() and np.isfinite(te).all():
                out[f"{Path(d).name}:{base}"] = (o, te)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", nargs="*", default=["ext/goodpjw"])
    ap.add_argument("--ours", nargs="*", default=["ebm_i30_s7_mib256", "ebm_i30_s7", "xgbt2_b2048",
                                                  "lm_distilgpt2_ep10"])
    ap.add_argument("--min_auc", type=float, default=0.944)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    mem = load_ext(a.ext, len(y), len(te_ids))
    for n in a.ours:
        mem[n] = (np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel(),
                  np.asarray(load_predictions(RESULTS_DIR, n, te_ids, split="test")).ravel())
    auc = {k: roc_auc_score(y, v[0]) for k, v in mem.items()}
    names = [k for k in sorted(auc, key=lambda k: -auc[k]) if auc[k] >= a.min_auc]
    for k in names:
        print(f"  {k:40s} {auc[k]:.5f}")
    Xo = np.column_stack([rank_logit(mem[k][0]) for k in names])
    Xt = np.column_stack([rank_logit(mem[k][1]) for k in names])

    def stack(cols: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
        return nested_lr_stack(Xo, y, Xt, c=a.C, cols=cols, max_iter=3000)

    ext_cols = [i for i, k in enumerate(names) if ":" in k]
    our_cols = [i for i, k in enumerate(names) if ":" not in k]
    print(f"ext only   {stack(ext_cols)[0]:.5f}")
    print(f"ours only  {stack(our_cols)[0]:.5f}")
    full_auc, oof, tst = stack(list(range(len(names))))
    print(f"ALL        {full_auc:.5f}")
    full = LogisticRegression(C=a.C, max_iter=3000).fit(Xo, y)
    print("  weights", {k: round(float(c), 3) for k, c in zip(names, full.coef_[0])})
    if a.out:
        prob = 1 / (1 + np.exp(-tst))
        save_predictions(RESULTS_DIR, a.out, oof=1 / (1 + np.exp(-oof)), test=prob, train_ids=ids,
                         test_ids=te_ids)
        save_cv_result(RESULTS_DIR, a.out, [], full_auc, metric_name="auc",
                       params={"members": names, "C": a.C})
        write_submission(SUBMISSIONS_DIR, f"{a.out}.csv", te_ids, TARGET, prob)


if __name__ == "__main__":
    main()
