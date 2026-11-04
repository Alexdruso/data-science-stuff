"""Pooled stack over every same-fold public OOF bank + our legs.

    python src/pool_stack.py --ext ext/dl ext/ko ... --ours lm_distilgpt2_ep10 [--out name]

Bank discovery (recursive): `<base>_oof.npy` with `<base>_{full,test}.npy`, or a directory
holding `oof.npy` + `test.npy` (masayakawamata format), or `oof_*.npy`/`test_*.npy` pairs.
Arrays must be n_train / n_test long, in id-sorted (= csv) order. Near-duplicates (rank corr
> --dedup) keep the higher-AUC member. Members enter as logit(rank); meta = L2 LR; the stack
score is outer StratifiedKFold(5, seed 7/13/99), which is nested w.r.t. the meta fit.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import load_predictions, save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def rl(v: np.ndarray) -> np.ndarray:
    r = rankdata(v) / (len(v) + 1)
    return np.log(r / (1 - r))


def _ok(o: np.ndarray, t: np.ndarray, n_tr: int, n_te: int) -> bool:
    return len(o) == n_tr and len(t) == n_te and bool(np.isfinite(o).all() and np.isfinite(t).all())


def load_banks(dirs: list[str], n_tr: int, n_te: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*.npy")):
            name, o_path, t_path = None, None, None
            if p.name.endswith("_oof.npy"):
                base = p.name[: -len("_oof.npy")]
                t_path = next((p.parent / f"{base}_{s}.npy" for s in ("full", "test")
                               if (p.parent / f"{base}_{s}.npy").exists()), None)
                name = f"{p.parent.name}:{base}"
            elif p.name.startswith("oof_"):
                base = p.name[len("oof_"):-4]
                t_path = next((p.parent / f"{s}_{base}.npy" for s in ("test", "refit", "foldavg")
                               if (p.parent / f"{s}_{base}.npy").exists()), None)
                name = f"{p.parent.name}:{base}"
            elif p.name == "oof.npy":
                t_path = p.parent / "test.npy"
                name = f"{p.parent.name}"
            if name is None or t_path is None or not t_path.exists():
                continue
            o_path = p
            try:
                o, t = np.load(o_path).astype(float), np.load(t_path).astype(float)
            except (ValueError, OSError):
                continue
            if o.ndim == 2 and o.shape[1] == 2:
                o, t = o[:, 1], t[:, 1] if t.ndim == 2 else t
            o, t = o.ravel(), t.ravel()
            if _ok(o, t, n_tr, n_te) and name not in out:
                out[name] = (o, t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", nargs="*", default=["ext"])
    ap.add_argument("--ours", nargs="*", default=["lm_distilgpt2_ep10"])
    ap.add_argument("--min_auc", type=float, default=0.946)
    ap.add_argument("--dedup", type=float, default=0.9995)
    ap.add_argument("--C", type=float, nargs="+", default=[1.0])
    ap.add_argument("--drop", nargs="*", default=[], help="substrings of member names to drop")
    ap.add_argument("--ablate", nargs="*", default=[], help="substrings; report stack without each")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    assert (np.diff(ids) > 0).all()
    mem = load_banks(a.ext, len(y), len(te_ids))
    for n in a.ours:
        mem[n] = (np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel(),
                  np.asarray(load_predictions(RESULTS_DIR, n, te_ids, split="test")).ravel())
    mem = {k: v for k, v in mem.items() if not any(s in k for s in a.drop)}
    auc = {k: roc_auc_score(y, v[0]) for k, v in mem.items()}
    names = [k for k in sorted(auc, key=lambda k: -auc[k]) if auc[k] >= a.min_auc]
    R = {k: rl(mem[k][0]) for k in names}
    kept: list[str] = []
    for k in names:
        dup = next((j for j in kept if np.corrcoef(R[k], R[j])[0, 1] > a.dedup), None)
        if dup is None:
            kept.append(k)
    print(f"{len(mem)} loaded, {len(names)} >= {a.min_auc}, {len(kept)} after dedup")
    for k in kept:
        print(f"  {k:60s} {auc[k]:.5f}")
    Xo = np.column_stack([R[k] for k in kept])
    Xt = np.column_stack([rl(mem[k][1]) for k in kept])

    def stack(cols: list[int], C: float) -> tuple[float, np.ndarray, np.ndarray]:
        oof, tst = np.zeros(len(y)), np.zeros(len(te_ids))
        for s in (7, 13, 99):
            for trn, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Xo, y):
                m = LogisticRegression(C=C, max_iter=5000).fit(Xo[trn][:, cols], y[trn])
                oof[va] += m.decision_function(Xo[va][:, cols]) / 3
                tst += m.decision_function(Xt[:, cols]) / 15
        return float(roc_auc_score(y, oof)), oof, tst

    allc = list(range(len(kept)))
    best = (-1.0, 0.0)
    for C in a.C:
        s = stack(allc, C)[0]
        print(f"ALL C={C:<8} {s:.6f}")
        best = max(best, (s, C))
    C = best[1]
    ours = [i for i, k in enumerate(kept) if ":" not in k and k in a.ours]
    if ours:
        print(f"without ours ({[kept[i] for i in ours]}): "
              f"{stack([i for i in allc if i not in ours], C)[0]:.6f}")
    for sub in a.ablate:
        cols = [i for i, k in enumerate(kept) if sub not in k]
        print(f"without *{sub}* ({len(allc) - len(cols)} members): {stack(cols, C)[0]:.6f}")
    full_auc, oof, tst = stack(allc, C)
    full = LogisticRegression(C=C, max_iter=5000).fit(Xo, y)
    w = sorted(zip(kept, full.coef_[0]), key=lambda t: -abs(t[1]))
    print("  top weights", {k: round(float(c), 3) for k, c in w[:15]})
    if a.out:
        prob = 1 / (1 + np.exp(-tst))
        save_predictions(RESULTS_DIR, a.out, oof=1 / (1 + np.exp(-oof)), test=prob, train_ids=ids,
                         test_ids=te_ids)
        save_cv_result(RESULTS_DIR, a.out, [], full_auc, metric_name="auc",
                       params={"members": kept, "C": C})
        write_submission(SUBMISSIONS_DIR, f"{a.out}.csv", te_ids, TARGET, prob)
        print(f"wrote {a.out}  OOF {full_auc:.6f}")


if __name__ == "__main__":
    main()
