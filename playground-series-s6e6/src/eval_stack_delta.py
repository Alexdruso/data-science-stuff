"""Non-destructive LR-stack delta gate (PS S6E6).

Mirrors build_lr_stack.py's exact LR config but takes a comma-separated MODELS list and
only PRINTS the OOF argmax/thresh — never writes oof/test/submission. Use to gate a new base
against the current 11-base stack (baseline / +new / swap) without clobbering production files.

Run:  python src/eval_stack_delta.py --models lgbm,xgboost,...   [--label baseline]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from postprocess import optimize_thresholds

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEEDS = [2024, 7, 13, 42, 99]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = LabelEncoder().fit_transform(tr[TARGET].to_numpy())

    Ztr, used = [], []
    for m in models:
        fo = RESULTS_DIR / f"oof_{m}.npy"
        if fo.exists():
            Ztr.append(logit(np.load(fo)))
            used.append(m)
        else:
            print(f"  !! missing oof_{m}.npy — skipped")
    Ztr = np.hstack(Ztr)

    oof = np.zeros((len(y), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Ztr, y):
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, n_jobs=-1)
            lr.fit(Ztr[tri], y[tri])
            oof[vai] += lr.predict_proba(Ztr[vai]) / len(SEEDS)

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    _, best = optimize_thresholds(oof, y)
    tag = f"[{args.label}] " if args.label else ""
    print(f"{tag}{len(used)} bases  argmax {argmax:.6f}  thresh {best:.6f}  "
          f"recall G/Q/S {rec[0]:.4f}/{rec[1]:.4f}/{rec[2]:.4f}")


if __name__ == "__main__":
    main()
