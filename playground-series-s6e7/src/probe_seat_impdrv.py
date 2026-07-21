"""Seat test for the driver-reconstruction canary (Day-17 P1 follow-up).

The imp arm replicated +0.0002/+0.0001 across two seeds — real but sub-bar
solo. The decision-relevant question is the SEAT: does the reconstruction
channel add anything the 8-leg stack lacks? Challenger OOF is assembled from
the probe's fold checkpoints (results/cache/ckpt_impdrv/imp_f*_val.npy,
seed-42 zoo split) — zero extra training. Standard 12-cell instrument.

  lr8  — incumbent lrstack leg set
  lri  — lr8 + impdrv-canary logits (hgbc_te3 + imp_sleep/sleep_completed)
  lrb  — lr8 + BASE-canary logits (control: same instrument, no imp cols —
         separates "9th hgbc leg" from "reconstruction channel")

Run: python src/probe_seat_impdrv.py -> results/probe_seat_impdrv.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from probe_seat_day17 import KEYS8, N_SPLITS, _arm
from train_common import N_CLASSES, N_FOLDS, RESULTS_DIR, load_dataset
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle.stacking import clipped_logit

SEED = 42


def cache_oof(arm: str, y: np.ndarray) -> np.ndarray:
    oof = np.zeros((len(y), N_CLASSES))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_tr, val_idx) in enumerate(skf.split(y, y), 1):
        oof[val_idx] = np.load(CACHE_DIR / "ckpt_impdrv" / f"{arm}_f{fold}_val.npy")
    return oof


def main() -> None:
    ds = load_dataset()
    y = ds.y
    X8 = np.hstack(
        [
            clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
            for k in KEYS8
        ]
    )
    mats = {"lr8": X8}
    for name, arm in (("lri", "imp"), ("lrb", "base")):
        mats[name] = np.hstack(
            [X8, clipped_logit(cache_oof(arm, y)).astype(np.float32)]
        )
    print({n: X.shape for n, X in mats.items()}, flush=True)

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    def cell(fit_idx, eval_idx):  # noqa: ANN001, ANN202
        return {name: _arm(X, y, fit_idx, eval_idx) for name, X in mats.items()}

    results = Parallel(n_jobs=4)(delayed(cell)(f, e) for f, e in cells)
    scores = {a: np.array([r[a] for r in results]) for a in results[0]}
    lines = [f"impdrv seat test, {len(cells)} cells; lr8={KEYS8}"]
    for a in sorted(scores, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    for a in ("lri", "lrb"):
        d = scores[a] - scores["lr8"]
        lines.append(f"per-cell {a}: " + " ".join(f"{x:+.5f}" for x in d))
    d_ib = scores["lri"] - scores["lrb"]
    lines.append(
        f"lri vs lrb (reconstruction channel net of 9th-leg effect): "
        f"{d_ib.mean():+.5f} (positive {int((d_ib > 0).sum())}/{len(d_ib)})"
    )
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_seat_impdrv.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
