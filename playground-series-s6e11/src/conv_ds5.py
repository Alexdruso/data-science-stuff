"""Convert 11-05 public datasets (csv/npz, id-keyed) to <name>_oof.npy / <name>_test.npy in id order."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
tr = pd.read_csv("data/train.csv", usecols=["id", "is_fraud"]).sort_values("id")
te_ids = np.sort(pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy())
tr_ids, y = tr["id"].to_numpy(), tr["is_fraud"].to_numpy()


def put(name: str, oid, o, tid, t) -> None:
    so = pd.Series(np.asarray(o, float), index=np.asarray(oid)).reindex(tr_ids).to_numpy()
    st = pd.Series(np.asarray(t, float), index=np.asarray(tid)).reindex(te_ids).to_numpy()
    if np.isnan(so).any() or np.isnan(st).any():
        print("SKIP nan", name, np.isnan(so).sum(), np.isnan(st).sum())
        return
    np.save(OUT / f"{name}_oof.npy", so)
    np.save(OUT / f"{name}_test.npy", st)
    print(f"{name:50s} {roc_auc_score(y, so):.5f}")


d = SRC / "ps-s6e11-tabpfn-gbdt-oof-test-pairs"
for f in sorted(d.glob("pair_oof_*.csv")):
    b = f.name[len("pair_oof_"):-4]
    o, t = pd.read_csv(f), pd.read_csv(d / f"pair_test_{b}.csv")
    put(f"sadam_{b}", o["id"], o["is_fraud"], t["id"], t["is_fraud"])
for f in ("predictions", "gen_predictions"):
    z = np.load(SRC / "fraud-shared-digit-cnn-weights" / f"{f}.npz")
    put(f"digitcnn_{f}", z["ids_train"], z["oof"], z["ids_test"], z["test"])
for f in sorted((SRC / "s6e11-tree-ensemble-predictions").glob("*.npz")):
    z = np.load(f)
    if "oof" in z.files and "fold_test" in z.files:
        put(f"wguesdon_{f.stem}", z["train_id"], z["oof"], z["test_id"], z["fold_test"].mean(0))
    else:
        print(f.stem, z.files)
d = SRC / "s6e11-cross-fitted-artifacts" / "fold_predictions"
for m in sorted({p.stem.rsplit("_fold", 1)[0] for p in d.glob("*.npz")}):
    o, t = np.full(len(tr_ids), np.nan), np.zeros(len(te_ids))
    for k in range(5):
        z = np.load(d / f"{m}_fold{k}.npz")
        o[z["valid_index"]] = z["valid"]
        t += z["test"] / 5
    put(f"mcf_{m}", tr_ids, o, te_ids, t)
z = np.load(SRC / "s6e11-digit-token-full-refit-artifacts" / "joint_token_predictions.npz")
put("mdt_joint_token", tr_ids, z["oof"], te_ids, z["test"])
