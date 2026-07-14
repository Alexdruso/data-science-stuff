"""Row-level memorization probe (Day-11, advisor-directed).

The 3-dim label-channel model (label independent of everything given the
stress x activity x exact-sleep triple) was established by an AGGREGATE statistic
scan. That scan is structurally blind to ROW-LEVEL memorization: a generator can
emit near-duplicate train/test blocks whose labels are copyable, and no aggregate
independence test would see it. The zoo has GBMs / MLPs / TabPFN / FTT but NEVER a
raw kNN base. Two falsifiable outcomes:
  * kNN OOF is competitive AND decorrelated -> a genuine new base -> seat-test it.
  * kNN is far below te13 and test->train distances show no near-zero spike ->
    memorization cleanly FALSIFIED (a real null, not the redundant residual probe).

Design: standardize the 7 numerics, one-hot the 6 cats (equal weight), 5-fold kNN
OOF balanced accuracy on a stratified subsample (fast, and near-duplicates survive
subsampling). Report BA + error-overlap vs the incumbent, and the test->train
nearest-neighbour distance distribution against full train (10k test sample).

Run: python src/probe_nn_memorization.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as ba
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
CL = ["at-risk", "fit", "unhealthy"]
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count",
       "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
SUB = 150_000
K = 25


def encode(df: pd.DataFrame, mu, sd, cat_levels) -> np.ndarray:  # noqa: ANN001
    z = (df[NUM].to_numpy(np.float64) - mu) / sd
    parts = [np.nan_to_num(z, nan=0.0)]  # NaN -> mean (0 after standardize)
    for c in CAT:
        lv = cat_levels[c]
        oh = np.zeros((len(df), len(lv)), np.float64)
        idx = df[c].astype(str).map({v: i for i, v in enumerate(lv)}).to_numpy()
        ok = ~pd.isna(idx)
        oh[np.where(ok)[0], idx[ok].astype(int)] = 1.0
        parts.append(oh)
    return np.hstack(parts)


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv").sort_values("id").reset_index(drop=True)
    te = pd.read_csv(DATA / "test.csv").sort_values("id").reset_index(drop=True)
    y = tr["health_condition"].map({c: i for i, c in enumerate(CL)}).to_numpy()
    for c in CAT:
        tr[c] = tr[c].astype(str)
        te[c] = te[c].astype(str)
    mu, sd = tr[NUM].mean().to_numpy(), tr[NUM].std().to_numpy() + 1e-9
    cat_levels = {c: sorted(set(tr[c]) | set(te[c])) for c in CAT}
    Xtr = encode(tr, mu, sd, cat_levels)
    Xte = encode(te, mu, sd, cat_levels)
    print(f"encoded train {Xtr.shape} test {Xte.shape}", flush=True)

    # incumbent errors for overlap
    oof_inc = np.load(RES / "oof_lrstack_u3_r.npy")
    err_inc = oof_inc.argmax(1) != y

    # --- kNN OOF on a stratified subsample ---
    rng = np.random.default_rng(42)
    sidx = np.concatenate([
        rng.permutation(np.where(y == c)[0])[: int(SUB * (y == c).mean())]
        for c in range(3)
    ])
    Xs, ys = Xtr[sidx], y[sidx]
    err_inc_s = err_inc[sidx]
    knn_pred = np.zeros(len(sidx), int)
    knn_proba = np.zeros((len(sidx), 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for f, (a, b) in enumerate(skf.split(Xs, ys), 1):
        nn = NearestNeighbors(n_neighbors=K, algorithm="auto", n_jobs=-1)
        nn.fit(Xs[a])
        _, ind = nn.kneighbors(Xs[b])
        votes = ys[a][ind]  # (nb, K)
        for c in range(3):
            knn_proba[b, c] = (votes == c).mean(1)
        knn_pred[b] = knn_proba[b].argmax(1)
        print(f"  fold {f} done", flush=True)
    knn_ba = ba(ys, knn_pred)
    np.save(RES / "_probe_knn_proba.npy", knn_proba)
    np.save(RES / "_probe_knn_sidx.npy", sidx)
    # direct blend check: does adding kNN proba lift the incumbent on the subsample?
    inc_s = oof_inc[sidx]
    print("\n-- blend check (argmax on inc + w*knn, subsample) --")
    base = ba(ys, inc_s.argmax(1))
    for w in (0.0, 0.05, 0.1, 0.2, 0.4):
        blended = inc_s + w * knn_proba
        print(f"  w={w:<4} BA={ba(ys, blended.argmax(1)):.4f}  (Δ={ba(ys, blended.argmax(1)) - base:+.4f})")
    err_knn = knn_pred != ys
    both = (err_knn & err_inc_s).sum()
    overlap = both / max(1, err_inc_s.sum())
    print(f"\nkNN(K={K}) OOF BA (subsample n={len(sidx):,}) = {knn_ba:.4f}")
    print(f"incumbent BA on same subsample        = {ba(ys, oof_inc.argmax(1)[sidx]):.4f}")
    print(f"kNN error rate {err_knn.mean():.4f} | incumbent {err_inc_s.mean():.4f}")
    print(f"error-overlap(kNN vs incumbent) = {overlap:.3f} "
          f"(low = decorrelated)")

    # --- test -> full-train nearest-neighbour distances (memorization detector) ---
    tq = rng.permutation(len(Xte))[:10_000]
    nn_full = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn_full.fit(Xtr)
    d, _ = nn_full.kneighbors(Xte[tq])
    d = d.ravel()
    # train self-distance baseline (nearest OTHER train row) on 10k sample
    tqs = rng.permutation(len(Xtr))[:10_000]
    d2, _ = nn_full.kneighbors(Xtr[tqs], n_neighbors=2)
    d_self = d2[:, 1]  # skip self
    print("\ntest->train NN distance   min/median/p05 = "
          f"{d.min():.3f}/{np.median(d):.3f}/{np.percentile(d, 5):.3f}")
    print("train->train NN distance  min/median/p05 = "
          f"{d_self.min():.3f}/{np.median(d_self):.3f}/{np.percentile(d_self, 5):.3f}")
    print(f"frac test rows with NN dist < 0.10 = {(d < 0.10).mean():.4f}")
    print(f"frac test rows with NN dist < 0.01 = {(d < 0.01).mean():.4f}")


if __name__ == "__main__":
    main()
