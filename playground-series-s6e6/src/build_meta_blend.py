"""Meta-of-metas probe — average the stack-level outputs (S6E6 Stage 1).

The LR stack (0.97055), GBDT stack, and alt-metas are all honest multi-seed outer-fold
OOF constructions over the SAME 10 bases, but with different meta-learner families
(linear / tree / ridge-cal-tree / MLP). Their residual errors differ slightly →
a simple prob-average or rank-average at the META level is the cheapest remaining
recombination. No fitting happens here (fixed uniform weights), so the OOF score of
the blend is as honest as its inputs.

GATE: threshold-tuned OOF balanced-acc must beat lrstack 0.97055 to be a candidate.

Run:  python src/build_meta_blend.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
METAS = ["lrstack", "gbdtstack", "altmeta_ridgecal-gbdt", "altmeta_nn"]
GATE = 0.97055
RUN = "metablend"


def rank_norm(p: np.ndarray) -> np.ndarray:
    """Per-class rank transform to [0,1] — robust to calibration differences."""
    out = np.empty_like(p)
    for c in range(p.shape[1]):
        out[:, c] = rankdata(p[:, c]) / len(p)
    return out


def main() -> None:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    oofs, tests, used = {}, {}, []
    for m in METAS:
        fo, ft = RESULTS_DIR / f"oof_{m}.npy", RESULTS_DIR / f"test_{m}.npy"
        if fo.exists() and ft.exists():
            oofs[m], tests[m] = np.load(fo), np.load(ft)
            used.append(m)
    print(f"Blending over metas: {used}  (GATE {GATE})")

    best = (None, None, -1.0, None, None)  # (name, members, thresh_score, oof, test)
    for r in range(2, len(used) + 1):
        for combo in itertools.combinations(used, r):
            for kind in ("avg", "rank"):
                if kind == "avg":
                    o = np.mean([oofs[m] for m in combo], axis=0)
                    t = np.mean([tests[m] for m in combo], axis=0)
                else:
                    o = np.mean([rank_norm(oofs[m]) for m in combo], axis=0)
                    t = np.mean([rank_norm(tests[m]) for m in combo], axis=0)
                argmax = float(balanced_accuracy_score(y, np.argmax(o, axis=1)))
                tw, th = optimize_thresholds(o, y)
                name = f"{kind}[{'+'.join(combo)}]"
                mark = "  <== BEAT GATE" if th > GATE else ""
                print(f"{name:65s} argmax {argmax:.5f}  thresh {th:.5f}{mark}")
                if th > best[2]:
                    best = (name, combo, th, o, t)

    name, combo, th, oof, test = best
    pred = np.argmax(oof, axis=1)
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nBEST: {name}  thresh {th:.5f}  vs GATE {GATE}")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")

    if th > GATE:
        tw, _ = optimize_thresholds(oof, y)
        save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
        save_cv_result(RESULTS_DIR, RUN, [], th, metric_name="balanced_acc")
        np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
        np.save(RESULTS_DIR / f"test_{RUN}.npy", test)
        labels = le.inverse_transform(np.argmax(test * tw, axis=1))
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
        print(f"Saved → {RUN} ({name})")
    else:
        print("No blend beat the gate — nothing saved.")


if __name__ == "__main__":
    main()
