"""Fast stacker experiments on a cached pool member matrix.

    python src/pool_lab.py cache --ext ... --ours ... [pool_stack filters]   -> results/pool_lab.npz
    python src/pool_lab.py run EXP [EXP ...]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CACHE = Path(__file__).parent.parent / "results" / "pool_lab.npz"


def cache() -> None:
    import pool_stack as ps

    captured = {}
    orig = ps.LogisticRegression

    def grab(*a, **k):  # stop pool_stack right after it builds the matrix
        raise SystemExit
    ps.LogisticRegression = grab
    import builtins
    real_stack = np.column_stack
    calls = []

    def cs(arrs):
        out = real_stack(arrs)
        calls.append(out)
        return out
    np.column_stack = cs
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    try:
        ps.main()
    except SystemExit:
        pass
    np.column_stack = real_stack
    ps.LogisticRegression = orig
    import polars as pl
    from features import TARGET, build_features
    y = build_features(pl.read_csv(ps.DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(int)
    np.savez(CACHE, Xo=calls[0], Xt=calls[1], y=y)
    print("cached", calls[0].shape, calls[1].shape)


def lr_stack(Xo, y, C=0.01, nest=5, seeds=(7, 13, 99), cols=None, Xt=None):
    cols = np.arange(Xo.shape[1]) if cols is None else cols
    oof = np.zeros(len(y))
    for s in seeds:
        for trn, va in StratifiedKFold(nest, shuffle=True, random_state=s).split(Xo, y):
            m = LogisticRegression(C=C, max_iter=5000).fit(Xo[trn][:, cols], y[trn])
            oof[va] += m.decision_function(Xo[va][:, cols]) / len(seeds)
    return roc_auc_score(y, oof), oof


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
            print("base C.01 nest5", lr_stack(Xo, y)[0], flush=True)
        elif exp == "nest10":
            print("nest10", lr_stack(Xo, y, nest=10)[0], flush=True)
        elif exp.startswith("C"):
            print(exp, lr_stack(Xo, y, C=float(exp[1:]))[0], flush=True)
        elif exp == "std":  # standardize columns before L2
            Xs = (Xo - Xo.mean(0)) / Xo.std(0)
            print("std C.01", lr_stack(Xs, y)[0], "C.001", lr_stack(Xs, y, C=0.001)[0], flush=True)
        elif exp == "subbag":  # random-subspace bag of LRs
            rng = np.random.default_rng(0)
            acc = np.zeros(len(y))
            for b in range(8):
                cols = np.sort(rng.choice(Xo.shape[1], Xo.shape[1] // 2, replace=False))
                s, o = lr_stack(Xo, y, cols=cols, seeds=(7,))
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
