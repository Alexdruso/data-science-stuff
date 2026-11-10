"""Price candidate members inside the cached pool (results/pool_lab.npz): stack AUC with each added.

    python src/pool_add.py nb_raw_m20 nb_dig_m20 [--together]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features

from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import nested_lr_stack, rank_logit

R = Path(__file__).parent.parent / "results"
z = np.load(R / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
ids = build_features(pl.read_csv(R.parent / "data" / "train.csv"))["id"].to_numpy()
names = [a for a in sys.argv[1:] if not a.startswith("--")]
cols = [rank_logit(np.asarray(load_predictions(R, n, ids)).ravel()) for n in names]
base = 0.9526657545740411  # cached v7 matrix, C .01 nest5 seeds 7/13/99
if "--together" in sys.argv:
    print("together", names, f"{nested_lr_stack(np.column_stack([Xo, *cols]), y).auc - base:+.6f}", flush=True)
else:
    for n, c in zip(names, cols):
        print(n, f"{nested_lr_stack(np.column_stack([Xo, c]), y).auc - base:+.6f}", flush=True)
