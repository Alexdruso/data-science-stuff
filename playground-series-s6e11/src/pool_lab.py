"""Fast stacker experiments on a cached pool member matrix.

    python src/pool_lab.py cache --ext ... --ours ... [pool_stack filters]   -> results/pool_lab.npz
    python src/pool_lab.py run EXP [EXP ...]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_science_stuff.kaggle.stacking import nested_lr_stack

CACHE = Path(__file__).parent.parent / "results" / "pool_lab.npz"


def cache() -> None:
    import pool_stack as ps

    pool = ps.build_pool(ps.parse_args(sys.argv[2:]))
    np.savez(CACHE, Xo=pool.Xo, Xt=pool.Xt, y=pool.y)
    print("cached", pool.Xo.shape, pool.Xt.shape)


def main() -> None:
    if sys.argv[1] == "cache":
        sys.path.insert(0, str(Path(__file__).parent))
        cache()
        return
    z = np.load(CACHE)
    Xo, y = z["Xo"], z["y"]
    print("matrix", Xo.shape, flush=True)
    for exp in sys.argv[2:]:
        if exp == "base":
            print("base C.01 nest5", nested_lr_stack(Xo, y).auc, flush=True)
        elif exp == "nest10":
            print("nest10", nested_lr_stack(Xo, y, n_splits=10).auc, flush=True)
        elif exp.startswith("C"):
            print(exp, nested_lr_stack(Xo, y, c=float(exp[1:])).auc, flush=True)
        elif exp == "std":  # standardize columns before L2
            Xs = (Xo - Xo.mean(0)) / Xo.std(0)
            print("std C.01", nested_lr_stack(Xs, y).auc, "C.001", nested_lr_stack(Xs, y, c=0.001).auc, flush=True)
        elif exp == "subbag":  # random-subspace bag of LRs
            rng = np.random.default_rng(0)
            acc = np.zeros(len(y))
            for b in range(8):
                cols = np.sort(rng.choice(Xo.shape[1], Xo.shape[1] // 2, replace=False))
                s, o, _ = nested_lr_stack(Xo, y, cols=cols, seeds=(7,))
                acc += o
                print(f"  subbag {b} {s:.6f} cum {roc_auc_score(y, acc):.6f}", flush=True)
        elif exp == "hill":  # greedy rank hill climb (with repeats), nested
            print("hill", hill(Xo, y), flush=True)


def hill(Xo, y, steps=40):
    R = np.argsort(np.argsort(Xo, 0), 0) / len(y)
    oof = np.zeros(len(y))
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=7).split(Xo, y):
        cur = np.zeros(len(trn)); w = np.zeros(Xo.shape[1]); best = 0.0
        for _ in range(steps):
            sc = [roc_auc_score(y[trn], cur + R[trn, j]) for j in range(Xo.shape[1])]
            j = int(np.argmax(sc))
            if sc[j] <= best + 1e-7:
                break
            best = sc[j]; cur += R[trn, j]; w[j] += 1
        oof[va] = R[va] @ w
    return roc_auc_score(y, oof)


if __name__ == "__main__":
    main()
