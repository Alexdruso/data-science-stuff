"""TabPFN-v2 missing-driver-region specialist — FEASIBILITY/PRICING probe (Day-7).

Not a gate run. Answers three questions only:
  1. Does TabPFN run at all on the 6 GB RTX 2060 with a 10k context?
  2. What does inference cost (sec/1k rows -> extrapolated to the test region)?
  3. Is its solo balanced accuracy on the missing-driver region even in the same
     league as the deployed core (oof_ensemble_r_breadth) on the SAME rows?

Region = rows with >=1 of the 3 key drivers missing (86% of residual error lives
there, headroom analysis). Run on the repaired surface to match the deployed
lineage. TabPFN context = 10k stratified region rows; eval = 20k disjoint region
rows (labels known, OOF-comparable against the core).

Run: S6E7_REPAIR=1 ../.venv/bin/python src/probe_tabpfn.py | tee results/probe_tabpfn.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, Dataset, load_dataset, robust_decision_weights

from data_science_stuff.kaggle_utils import weighted_predict

KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
N_CONTEXT = 10_000
N_EVAL = 20_000
CORE_KEY = "ensemble_r_breadth"


def encode(frame: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Ordinal-encode categoricals, keep NaN as NaN (TabPFN handles missingness)."""
    out = frame.copy()
    for c in cat_cols:
        codes = out[c].astype("category").cat.codes.astype("float64")
        out[c] = codes.replace(-1.0, np.nan)
    return out.to_numpy(dtype=np.float64)


def main() -> None:
    import torch
    from tabpfn import TabPFNClassifier

    ds: Dataset = load_dataset()
    region = ds.train[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    print(f"missing-driver region: {region.sum():,} train rows ({region.mean():.1%})")

    rng = np.random.default_rng(0)
    idx = rng.permutation(np.flatnonzero(region))
    ctx_idx, eval_idx = idx[:N_CONTEXT], idx[N_CONTEXT : N_CONTEXT + N_EVAL]

    X = encode(ds.train, ds.cat_cols)
    clf = TabPFNClassifier(device="cuda")
    t0 = time.time()
    clf.fit(X[ctx_idx], ds.y[ctx_idx])
    t_fit = time.time() - t0
    t0 = time.time()
    proba = clf.predict_proba(X[eval_idx])
    t_pred = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    del clf
    torch.cuda.empty_cache()

    sec_per_1k = t_pred / (N_EVAL / 1000)
    test_region_est = 296_000 * region.mean() * sec_per_1k / 1000
    print(f"fit {t_fit:.1f}s | predict {N_EVAL:,} rows in {t_pred:.1f}s "
          f"({sec_per_1k:.2f}s/1k) | peak VRAM {vram:.2f} GiB")
    print(f"extrapolated test-region inference (~{region.mean():.0%} of 296k): "
          f"~{test_region_est * 1000:.0f}s per context; OOF coverage of the train "
          f"region needs {region.sum() / 1000:.0f}k rows -> "
          f"~{region.sum() * sec_per_1k / 1000 / 60:.0f} min per context")

    y_ev = ds.y[eval_idx]
    core = np.load(RESULTS_DIR / f"oof_{CORE_KEY}.npy")[eval_idx]
    rows = [
        ("tabpfn argmax", proba.argmax(1)),
        ("core argmax (deployed)", core.argmax(1)),
    ]
    # in-sample weighted upper bounds, same treatment both sides
    for name, pr in [("tabpfn", proba), ("core", core)]:
        w = robust_decision_weights(y_ev, pr)
        rows.append((f"{name} weighted (in-sample UB)", weighted_predict(pr, w)))
    print(f"\n{'predictor':<28} {'bacc':>7}  per-class recall")
    for name, pred in rows:
        b = balanced_accuracy_score(y_ev, pred)
        rec = [float((pred[y_ev == c] == c).mean()) for c in range(3)]
        print(f"{name:<28} {b:>7.4f}  {np.round(rec, 3)}")


if __name__ == "__main__":
    main()
