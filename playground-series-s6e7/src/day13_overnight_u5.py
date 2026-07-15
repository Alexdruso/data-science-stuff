"""Day-13 overnight tail: 12-seed realmlp breadth (DISTINCT key) -> lrstack_u5.

Runs after the seeds 101/202/303/404 chain. Writes realmlp_te13_u_breadth12
(never touches the u4-audited breadth files), then builds lrstack_u5 with it.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import json

import numpy as np

from train_common import RESULTS_DIR, load_dataset, robust_decision_weights
from features import CLASSES
from data_science_stuff.kaggle_utils import weighted_predict
from sklearn.metrics import balanced_accuracy_score

SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777, 101, 202, 303, 404]

ds = load_dataset()
oofs, tests, used = [], [], []
for s in SEEDS:
    op = RESULTS_DIR / f"oof_realmlp_te13_u_s{s}.npy"
    tp = RESULTS_DIR / f"test_realmlp_te13_u_s{s}.npy"
    if op.exists() and tp.exists():
        oofs.append(np.load(op)); tests.append(np.load(tp)); used.append(s)
assert len(used) >= 10, f"only {used} present — seeds chain incomplete"
oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
np.save(RESULTS_DIR / "oof_realmlp_te13_u_breadth12.npy", oof)
np.save(RESULTS_DIR / "test_realmlp_te13_u_breadth12.npy", test)
dw = robust_decision_weights(ds.y, oof)
json.dump(dict(zip(CLASSES, dw.tolist())),
          open(RESULTS_DIR / "decision_weights_realmlp_te13_u_breadth12.json", "w"))
print(f"breadth12: {len(used)} seeds {used} -> weighted "
      f"{balanced_accuracy_score(ds.y, weighted_predict(oof, dw)):.4f}")

env = {**os.environ, "S6E7_STACK_NAME": "lrstack_u5", "S6E7_STACK_VARIANTS": "raw",
       "S6E7_STACK_SEEDS": "2024,7,13,42,99,1,2,3,4,5"}
subprocess.run(
    [sys.executable, str(Path(__file__).parent / "build_lr_stack.py"),
     "lgbm_bagw", "hgbc_bagw", "xgboost_bagw", "catboost_bagw", "mlp",
     "hgbc_te3_u_breadth", "lgbm_te3_u_breadth", "realmlp_te13_u_breadth12"],
    env=env, check=True)
