"""Grid-search Deotte threshold values using 10-fold stratified CV.

Fixed strategy: keep Deotte's fitted COEFS/INTERCEPTS, only vary the 4 numeric
thresholds that define the binary input features. The categorical columns are
precomputed once and concatenated per trial to keep the loop fast.

Baseline: soil=25, temp=30, rain=300, wind=10  (Deotte originals)
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from cv_results import save_cv_result
from formula import CLASSES, COEFS, INTERCEPTS

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 10

# Search grid — centred on Deotte originals, wider for Rainfall_mm
# (error analysis showed it has the highest boundary-error rate)
SOIL_GRID = np.arange(18.0, 33.0, 1.0)      # 15 values  [18 .. 32]
TEMP_GRID = np.arange(26.0, 36.0, 1.0)      # 10 values  [26 .. 35]
RAIN_GRID = np.arange(200.0, 410.0, 10.0)   # 21 values  [200 .. 400]
WIND_GRID = np.arange(6.0, 16.0, 1.0)       # 10 values  [6 .. 15]

TOTAL_COMBOS = len(SOIL_GRID) * len(TEMP_GRID) * len(RAIN_GRID) * len(WIND_GRID)


def _softmax_argmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).argmax(axis=1)


def main() -> None:
    print("Loading data …")
    train = pl.read_csv(DATA_DIR / "train.csv")
    n = len(train)

    # Target — encode to match CLASSES order: Low=0, Medium=1, High=2
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([cls_to_idx[v] for v in train["Irrigation_Need"].to_list()], dtype=np.int8)

    # Precompute fixed (categorical) binary columns — these never change
    cgs = train["Crop_Growth_Stage"].to_numpy()
    mulch = train["Mulching_Used"].to_numpy()
    fixed_cols = np.column_stack([
        (cgs == "Flowering").astype(np.int8),
        (cgs == "Harvest").astype(np.int8),
        (cgs == "Sowing").astype(np.int8),
        (cgs == "Vegetative").astype(np.int8),
        (mulch == "No").astype(np.int8),
        (mulch == "Yes").astype(np.int8),
    ])  # shape (n, 6)

    # Precompute raw numeric arrays (avoid repeated .to_numpy() in loop)
    soil_raw = train["Soil_Moisture"].to_numpy()
    temp_raw = train["Temperature_C"].to_numpy()
    rain_raw = train["Rainfall_mm"].to_numpy()
    wind_raw = train["Wind_Speed_kmh"].to_numpy()

    # Precompute fold indices
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n), y))

    # Baseline (Deotte originals)
    baseline_X = np.column_stack([
        (soil_raw < 25).astype(np.int8),
        (temp_raw > 30).astype(np.int8),
        (rain_raw < 300).astype(np.int8),
        (wind_raw > 10).astype(np.int8),
        fixed_cols,
    ])
    baseline_logits = INTERCEPTS + baseline_X @ COEFS
    baseline_preds = _softmax_argmax(baseline_logits)
    baseline_acc = float((baseline_preds == y).mean())
    print(f"Baseline accuracy (Deotte originals): {baseline_acc:.6f}")
    print(f"Grid size: {TOTAL_COMBOS:,} combinations × {N_FOLDS} folds\n")

    best_acc = 0.0
    best_params: tuple[float, float, float, float] = (25.0, 30.0, 300.0, 10.0)
    best_fold_accs: list[float] = []
    results: list[tuple[float, float, float, float, float]] = []

    t0 = time.time()
    for i, (soil_t, temp_t, rain_t, wind_t) in enumerate(
        itertools.product(SOIL_GRID, TEMP_GRID, RAIN_GRID, WIND_GRID), 1
    ):
        X = np.column_stack([
            (soil_raw < soil_t).astype(np.int8),
            (temp_raw > temp_t).astype(np.int8),
            (rain_raw < rain_t).astype(np.int8),
            (wind_raw > wind_t).astype(np.int8),
            fixed_cols,
        ])
        logits = INTERCEPTS + X @ COEFS
        preds = _softmax_argmax(logits)

        fold_accs = [
            float((preds[val_idx] == y[val_idx]).mean())
            for _, val_idx in folds
        ]
        oof_acc = float(np.mean(fold_accs))
        results.append((oof_acc, soil_t, temp_t, rain_t, wind_t))

        if oof_acc > best_acc:
            best_acc = oof_acc
            best_params = (soil_t, temp_t, rain_t, wind_t)
            best_fold_accs = fold_accs

        if i % 5000 == 0 or i == TOTAL_COMBOS:
            elapsed = time.time() - t0
            eta = elapsed / i * (TOTAL_COMBOS - i)
            print(
                f"  [{i:>6}/{TOTAL_COMBOS}]  best so far: {best_acc:.6f}  "
                f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed:.1f}s")

    # Sort and print top 10
    results.sort(reverse=True)
    print(f"\n{'Rank':<5} {'OOF Acc':>10} {'Soil<':>7} {'Temp>':>7} {'Rain<':>7} {'Wind>':>7}")
    for rank, (acc, s, t, r, w) in enumerate(results[:10], 1):
        marker = " ← best" if rank == 1 else ""
        print(f"{rank:<5} {acc:>10.6f} {s:>7.1f} {t:>7.1f} {r:>7.1f} {w:>7.1f}{marker}")

    soil_t, temp_t, rain_t, wind_t = best_params
    delta = best_acc - baseline_acc
    print(f"\nBest thresholds : Soil_Moisture < {soil_t}  |  Temperature_C > {temp_t}  |  Rainfall_mm < {rain_t}  |  Wind_Speed_kmh > {wind_t}")
    print(f"Best OOF acc    : {best_acc:.6f}  (Δ {delta:+.6f} vs Deotte baseline)")

    # Save to cv_scores
    save_cv_result(RESULTS_DIR, "tuned_thresholds", best_fold_accs, best_acc)

    # Save full results table
    import pandas as pd
    df_out = pd.DataFrame(results, columns=["oof_acc", "soil_thresh", "temp_thresh", "rain_thresh", "wind_thresh"])
    out_path = RESULTS_DIR / "threshold_search.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Full results saved → {out_path}")


if __name__ == "__main__":
    main()
