"""Normalise every public OOF/test source into one pool directory (PR-21 track).

    python src/pool_build.py

Writes data/public_oof/_pool/{oof,test}_<src>__<name>.npy (logits, float32, id order)
for every pair that passes the checks below; prints one line per accepted / rejected pair.

Checks per pair: lengths, no NaN; OOF AUC in [0.90, 0.9635] (above that is an in-sample
"OOF"); test aligned with id order (Spearman with the reference test leg ≥ 0.9); and not a
near-copy of a pair already accepted (OOF corr > 0.99999). Pairs on other fold seeds are
kept: each row is still out of fold under its own split, which is all a stacker needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
PUB = DATA_DIR / "public_oof"
OUT = PUB / "_pool"
N_TR, N_TE = 699_635, 299_844
SKIP_COLS = {"id", "fold", "satisfaction", "y", "target"}
# Stack-level members (blends/stacks fit on their own folds) leak into a nested stack.
SKIP_NAMES = ("nj_", "blend", "stack")


def logit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.min() < 0 or a.max() > 1:
        return a
    q = np.clip(a, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def sources() -> list[tuple[str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    # 1. oof_*.npy / test_*.npy directories
    for d in sorted(p for p in PUB.iterdir() if p.is_dir() and p.name != "_pool"):
        tag = d.name.replace("s6e10-", "").replace("ps-", "")[:14]
        for f in sorted(d.glob("oof_*.npy")):
            t = f.with_name(f.name.replace("oof_", "test_", 1))
            if t.exists():
                out.append((f"{tag}__{f.stem[4:]}", np.load(f), np.load(t)))
        # 2. npz with oof/test keys
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "oof" in z.files and "test" in z.files:
                out.append((f"{tag}__{f.stem}", z["oof"], z["test"]))
        # 3. oof/test csv pairs (id, pred): oof_<m>.csv or <m>_oof.csv
        for f in sorted([*d.glob("oof_*.csv"), *d.glob("*_oof.csv")]):
            if f.name.startswith("oof_"):
                t, stem = f.with_name(f.name.replace("oof_", "test_", 1)), f.stem[4:]
            else:
                t, stem = f.with_name(f.name.replace("_oof.csv", "_test.csv")), f.stem[:-4]
            if t.exists():
                o, tt = pd.read_csv(f).sort_values("id"), pd.read_csv(t).sort_values("id")
                out.append((f"{tag}__{stem}", o.iloc[:, -1].to_numpy(), tt.iloc[:, -1].to_numpy()))
        # 4. wide parquet pairs
        for f in sorted(d.glob("oof*.parquet")):
            t = next((g for g in d.glob("test*.parquet")), None)
            if t is None:
                continue
            o, tt = pd.read_parquet(f).sort_values("id"), pd.read_parquet(t).sort_values("id")
            for c in o.columns:
                if c not in SKIP_COLS and c in tt.columns:
                    out.append((f"{tag}__{c}", o[c].to_numpy(), tt[c].to_numpy()))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.npy"):  # members dropped by this build must not linger
        f.unlink()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    ref_t = logit(np.load(RESULTS_DIR / "test_lgbm_pr12_route_og_agg_orig.npy"))
    kept_o: list[np.ndarray] = []
    n_ok = 0
    for name, o, t in sources():
        if any(k in name.split("__", 1)[1] for k in SKIP_NAMES):
            print(f"REJECT stack-level  {name}")
            continue
        o, t = np.asarray(o, dtype=np.float64).ravel(), np.asarray(t, dtype=np.float64).ravel()
        if len(o) != N_TR or len(t) != N_TE or np.isnan(o).any() or np.isnan(t).any():
            print(f"REJECT shape/nan  {name} {len(o)} {len(t)}")
            continue
        lo, lt = logit(o), logit(t)
        auc = roc_auc_score(y, lo)
        if not 0.90 <= auc <= 0.9635:
            print(f"REJECT auc {auc:.6f}  {name}")
            continue
        rho = spearmanr(lt[::50], ref_t[::50]).statistic
        if rho < 0.9:
            print(f"REJECT test-align rho {rho:.3f}  {name}")
            continue
        z = (lo - lo.mean()) / lo.std()
        if any(float(np.mean(z * k)) > 0.99999 for k in kept_o):
            print(f"dup  {name}")
            continue
        kept_o.append(z)
        np.save(OUT / f"oof_{name}.npy", lo.astype(np.float32))
        np.save(OUT / f"test_{name}.npy", lt.astype(np.float32))
        n_ok += 1
        print(f"ok  {auc:.6f}  {name}")
    print(f"{n_ok} pairs in {OUT}")


if __name__ == "__main__":
    main()
