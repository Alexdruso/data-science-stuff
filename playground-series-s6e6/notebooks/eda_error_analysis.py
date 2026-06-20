"""Error-focused EDA — where do the remaining OOF errors live? (throwaway, 2026-06-10)

Questions:
 1. Confusion structure of the current best stack OOF (lrstack, 11-base).
 2. Error localisation in redshift space.
 3. Shared-vs-model-specific errors across the 11 bases (noise floor or recoverable?).
 4. Synthetic-data artifacts: exact duplicate feature rows (train/train+test) and
    label consistency among duplicates; repeated exact float values per column.
 5. Margin distribution — how many rows sit on a knife edge?
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import TARGET, build_features
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

DATA = Path(__file__).parent.parent / "data"
RES = Path(__file__).parent.parent / "results"

tr = build_features(pl.read_csv(DATA / "train.csv"))
te = build_features(pl.read_csv(DATA / "test.csv"))
le = LabelEncoder()
y = le.fit_transform(tr[TARGET].to_numpy())
classes = le.classes_.tolist()
print(f"classes {classes}")

oof = np.load(RES / "oof_lrstack.npy")
pred = np.argmax(oof, axis=1)
err = pred != y
print(f"\n=== 1. lrstack OOF errors: {err.sum()} / {len(y)} ({err.mean():.4%}) ===")
cm = confusion_matrix(y, pred)
print(pl.DataFrame({"true": classes} | {c: cm[:, i] for i, c in enumerate(classes)}))

z = tr["redshift"].to_numpy()
print("\n=== 2. error rate by redshift bin ===")
bins = [-1, 0.002, 0.05, 0.15, 0.5, 1.0, 2.0, 8.0]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (z > lo) & (z <= hi)
    if m.sum() == 0:
        continue
    e = err[m]
    share = m.sum() / len(y)
    print(
        f"z in ({lo:>6}, {hi:>5}]  rows {m.sum():>7} ({share:5.1%})  "
        f"err {e.mean():8.4%}  err-share {e.sum() / err.sum():6.1%}"
    )

print("\n=== 3. shared vs model-specific errors across bases ===")
bases = ["lgbm", "xgboost", "catboost", "lgbm_fe", "xgb_deotte", "realmlp_deotte",
         "catboost_deotte", "catboost_v3", "xgb_v3fe", "chain_cascade", "chain_cascade_xgb"]
errs = []
for b in bases:
    p = np.argmax(np.load(RES / f"oof_{b}.npy"), axis=1)
    errs.append(p != y)
E = np.stack(errs)  # (11, n)
n_wrong = E.sum(axis=0)
print("rows by #bases wrong (of 11):")
for k in range(12):
    c = int((n_wrong == k).sum())
    if c:
        print(f"  {k:>2} bases wrong: {c:>7} rows ({c / len(y):6.2%})")
all_wrong = n_wrong == 11
print(f"\nstack errors where ALL 11 bases wrong: {(err & all_wrong).sum()} / {err.sum()} "
      f"({(err & all_wrong).sum() / err.sum():.1%})  <- noise-floor share")
some_right = err & (n_wrong <= 8)
print(f"stack errors where >=3 bases RIGHT: {some_right.sum()} ({some_right.sum() / err.sum():.1%})"
      "  <- recoverable by better meta")

print("\n=== 4. duplicates & repeated exact values ===")
featcols = [c for c in tr.columns if c not in (TARGET, "id")]
rawcols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
           "spectral_type", "galaxy_population"]
dup_tr = tr.group_by(rawcols).agg(pl.len().alias("n"), pl.col(TARGET).n_unique().alias("nlab"))
multi = dup_tr.filter(pl.col("n") > 1)
print(f"train exact-dup feature rows: {multi['n'].sum() if len(multi) else 0} rows in "
      f"{len(multi)} groups; groups with conflicting labels: "
      f"{len(multi.filter(pl.col('nlab') > 1))}")
trte = pl.concat([tr.select(rawcols), te.select(rawcols)])
dup_all = trte.group_by(rawcols).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print(f"train+test exact-dup rows: {dup_all['n'].sum() if len(dup_all) else 0} in {len(dup_all)} groups")
print("\nrepeated exact values per raw float col (top share of rows in non-unique values):")
for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
    vc = trte[c].value_counts().filter(pl.col("count") > 1)
    rep = vc["count"].sum() if len(vc) else 0
    print(f"  {c:>9}: {rep / len(trte):7.2%} of rows share an exact value "
          f"({len(vc)} distinct repeated values)")

print("\n=== 5. margin distribution (top1 - top2 prob) on stack OOF ===")
s = np.sort(oof, axis=1)
margin = s[:, -1] - s[:, -2]
for thr in (0.02, 0.05, 0.1, 0.2):
    m = margin < thr
    print(f"margin < {thr:4}: {m.sum():>7} rows ({m.mean():6.2%})  "
          f"err rate inside {err[m].mean():7.2%}  share of all errors {err[m].sum() / err.sum():6.1%}")

print("\n=== STAR/GAL confusion zone detail (z <= 0.15) ===")
m = z <= 0.15
sub_err = err & m
ct = pl.DataFrame({
    "true": [classes[i] for i in y[m]],
    "pred": [classes[i] for i in pred[m]],
}).group_by(["true", "pred"]).len().sort("len", descending=True)
print(ct.head(9))
