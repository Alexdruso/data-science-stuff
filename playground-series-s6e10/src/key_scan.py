"""Hidden-key scan (10-29): does the pr42 stack's residual cluster by any column key?

    python src/key_scan.py [--stack pr42_lr_allown_c01_n10] [--jobs 4]

For each key (single raw columns, all pairs, a few wide combos): out-of-fold shrunk mean of the
stack's residual y - sigmoid(s) per key value (5-fold, seed 11; a row never sees its own residual),
then 5-fold LR on (s, key_te) minus LR on s alone (seed 7). Controls: noise column, and shuffled
versions of the widest keys (same cardinality). Prints coverage (share of rows whose key value
also occurs in other folds) and the delta, sorted.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

RAW = ["age", "flight_distance", "inflight_wifi_service", "departure_arrival_time_convenient",
       "ease_of_online_booking", "gate_location", "food_and_drink", "online_boarding", "seat_comfort",
       "inflight_entertainment", "on_board_service", "leg_room_service", "baggage_handling",
       "checkin_service", "cleanliness", "departure_delay_in_minutes", "arrival_delay_in_minutes",
       "gender", "customer_type", "type_of_travel", "class"]
RAT = RAW[2:15]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="pr42_lr_allown_c01_n10")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--k", type=float, default=30.0)
    a = ap.parse_args()
    d = build_features(pl.read_csv("data/train.csv"))
    y = d[TARGET].to_numpy().astype(float)
    s = np.load(f"results/oof_{a.stack}.npy")
    r = y - expit(s)
    codes = {c: d[c].fill_null(-1).rank("dense").to_numpy().astype(np.int64) for c in RAW}
    te_folds = list(StratifiedKFold(5, shuffle=True, random_state=11).split(s, y))
    lr_folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(s, y))

    def combo(cols: tuple[str, ...]) -> np.ndarray:
        k = np.zeros(len(y), np.int64)
        for c in cols:
            k = k * (codes[c].max() + 1) + codes[c]
        return np.unique(k, return_inverse=True)[1]

    keys: dict[str, np.ndarray] = {}
    for c in RAW:
        keys[c] = combo((c,))
    for c1, c2 in itertools.combinations(RAW, 2):
        keys[f"{c1}*{c2}"] = combo((c1, c2))
    wide = {
        "ratings14": tuple(RAT),
        "ratings14+cats": (*RAT, "gender", "customer_type", "type_of_travel", "class"),
        "ratings14+age": (*RAT, "age"),
        "ratings14+fd": (*RAT, "flight_distance"),
        "fd*age*class": ("flight_distance", "age", "class"),
        "fd*age*cats": ("flight_distance", "age", "gender", "customer_type", "type_of_travel", "class"),
        "fd*delays": ("flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes"),
        "age*delays": ("age", "departure_delay_in_minutes", "arrival_delay_in_minutes"),
        "delays*cats": ("departure_delay_in_minutes", "arrival_delay_in_minutes", "gender",
                        "customer_type", "type_of_travel", "class"),
        "all_but_delays": tuple(RAW[:15] + RAW[17:]),
        "all": tuple(RAW),
    }
    for n, cols in wide.items():
        keys[n] = combo(cols)
    rng = np.random.default_rng(0)
    for n in ["flight_distance", "fd*age*class", "ratings14", "fd*age*cats"]:
        keys[f"SHUF:{n}"] = rng.permutation(keys[n])

    def lr_auc(X: np.ndarray) -> float:
        o = np.zeros(len(y))
        for tr, va in lr_folds:
            o[va] = LogisticRegression(max_iter=300).fit(X[tr], y[tr]).decision_function(X[va])
        return roc_auc_score(y, o)

    base = lr_auc(s[:, None])

    def screen(name: str, key: np.ndarray) -> tuple[str, int, float, float]:
        te = np.zeros(len(y))
        cov = 0
        m = key.max() + 1
        for tr, va in te_folds:
            sm = np.bincount(key[tr], r[tr], m)
            ct = np.bincount(key[tr], minlength=m)
            te[va] = sm[key[va]] / (ct[key[va]] + a.k)
            cov += int((ct[key[va]] > 0).sum())
        if te.std() == 0:
            return name, m, cov / len(y), 0.0
        return name, m, cov / len(y), lr_auc(np.c_[s, te / te.std()]) - base

    print(f"base {base:.6f}; {len(keys)} keys", flush=True)
    noise = rng.normal(size=len(y))
    print(f"noise column delta {lr_auc(np.c_[s, noise]) - base:+.6f}", flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(screen)(n, k) for n, k in keys.items())
    for n, m, cov, dlt in sorted(res, key=lambda t: -t[3]):
        print(f"{dlt:+.6f}  card {m:7d}  cov {cov:.3f}  {n}", flush=True)


if __name__ == "__main__":
    main()
