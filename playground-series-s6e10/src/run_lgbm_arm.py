"""Paired LGBM feature arms for PS S6E10 (PR-3, PR-4).

    python src/run_lgbm_arm.py --arm base        # must reproduce lgbm_pr1_base
    python src/run_lgbm_arm.py --arm orig        # PR-3: original as a feature source
    python src/run_lgbm_arm.py --arm te_pairs    # PR-4a: fold-safe TE on rating pairs
    python src/run_lgbm_arm.py --arm cat_ratings # PR-4b: ratings as categoricals
    python src/run_lgbm_arm.py --arm profile     # PR-10: per-row rating profile
    python src/run_lgbm_arm.py --arm segment     # PR-10: deviation from segment means
    python src/run_lgbm_arm.py --arm numcat      # PR-10: numerics as category + counts

Every arm uses the PR-1 control's exact LGBM config (CPU, 8 threads, seed 42,
same folds), so its per-fold AUC is paired against `oof_lgbm_pr1_base.npy`.

Arms:
  orig        — no synthetic labels are used, so these are global features
                (leakage rule 2 territory, documented in CLAUDE.md):
                * `orig_pred`: an LGBM fit on the full ORIGINAL dataset, predicting
                  every synthetic train/test row;
                * `orig_rate_<col>`: the original's satisfaction rate for the row's
                  value of each categorical / rating column.
  te_pairs    — for each of the 78 rating pairs, a smoothed target rate of the
                (r_i, r_j) combination, fit INSIDE each fold on the training split
                (nested CV on train, `kaggle.encoding.add_fold_safe_target_encoding`).
  cat_ratings — the 13 0-5 ratings as `category` dtype (0 = N/A is not ordinal).
"""

from __future__ import annotations

import argparse
import os
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import RATING_COLS, TARGET
from matrices import ORIGINAL_FLAG, Matrices, build_matrices, load_original
from train_lgbm_orig import PARAMS

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.encoding import add_fold_safe_target_encoding
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
CONTROL = "lgbm_pr1_base"
ROUTE_CONTROL = "lgbm_arm_route"
TE_SMOOTH: float | str = float(os.environ.get("S6E10_TE_SMOOTH", "20")) if os.environ.get("S6E10_TE_SMOOTH", "20") != "auto" else "auto"
BAR = 0.0005

RATING_PAIRS = list(itertools.combinations(RATING_COLS, 2))
PAIR_COLS = [f"pair_{a}__{b}" for a, b in RATING_PAIRS]


def add_orig_features(m: Matrices) -> None:
    """PR-3: features computed from the original dataset only (no synthetic labels)."""
    assert m.X_orig is not None and m.y_orig is not None
    cols = [c for c in m.feature_cols if c != ORIGINAL_FLAG]
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=8,
    )
    model.fit(m.X_orig[cols], m.y_orig)
    orig = load_original().to_pandas()
    prior = float(m.y_orig.mean())
    for frame in (m.X, m.X_test):
        frame["orig_pred"] = model.predict_proba(frame[cols])[:, 1]
        for col in [*m.cat_cols, *RATING_COLS]:
            rates = orig.groupby(orig[col].astype(str))[TARGET].mean()
            frame[f"orig_rate_{col}"] = (
                frame[col].astype(str).map(rates).astype(float).fillna(prior)
            )


def add_pair_keys(m: Matrices) -> None:
    """PR-4a: integer key per rating pair (TE is fit later, inside each fold)."""
    for frame in (m.X, m.X_test):
        for (a, b), name in zip(RATING_PAIRS, PAIR_COLS):
            frame[name] = frame[a].astype(np.int16) * 10 + frame[b].astype(np.int16)


NUMCAT_COLS = ["age", "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes"]


def add_profile(m: Matrices) -> None:
    """PR-10 profile: dispersion, within-row z-scores, 0..5 histogram."""
    for frame in (m.X, m.X_test):
        r = frame[RATING_COLS].to_numpy(dtype=np.float32)
        mean, std = r.mean(axis=1, keepdims=True), r.std(axis=1, keepdims=True)
        frame["rating_std"] = std[:, 0]
        frame["rating_median"] = np.median(r, axis=1)
        z = (r - mean) / np.where(std > 0, std, 1.0)
        for i, col in enumerate(RATING_COLS):
            frame[f"z_{col}"] = z[:, i]
        for v in range(6):
            frame[f"n_rating_{v}"] = (r == v).sum(axis=1).astype(np.int8)


def add_segment(m: Matrices) -> None:
    """PR-10 segment: rating minus its (travel x class) segment mean (no labels)."""
    both = pd.concat([m.X, m.X_test])
    key = both["type_of_travel"].astype(str) + "|" + both["class"].astype(str)
    means = both[RATING_COLS].groupby(key.to_numpy()).mean()
    codes = {k: i for i, k in enumerate(sorted(means.index))}
    for frame in (m.X, m.X_test):
        fk = (frame["type_of_travel"].astype(str) + "|" + frame["class"].astype(str)).to_numpy()
        frame["segment"] = pd.Categorical(pd.Series(fk).map(codes).to_numpy(), categories=list(range(len(codes))))
        seg_means = means.loc[fk].to_numpy()
        dev = frame[RATING_COLS].to_numpy(dtype=np.float32) - seg_means
        for i, col in enumerate(RATING_COLS):
            frame[f"segdev_{col}"] = dev[:, i]
        frame["segdev_mean"] = dev.mean(axis=1)


def add_numcat(
    m: Matrices, cols: list[str] = NUMCAT_COLS, cnt: bool = True, cat: bool = True
) -> None:
    """PR-10 numcat: numerics duplicated as category + per-value frequency."""
    both = pd.concat([m.X, m.X_test])
    for col in cols:
        counts = both[col].value_counts(dropna=False)
        levels = counts.index[counts >= 20].dropna()
        for frame in (m.X, m.X_test):
            if cnt:
                frame[f"cnt_{col}"] = frame[col].map(counts).astype(float).to_numpy()
            if cat:
                frame[f"cat_{col}"] = pd.Categorical(frame[col].where(frame[col].isin(levels)), categories=levels)


TE_FD_COLS = ["te_fd", "te_fd_class", "te_fd_travel", "te_fd_cust"]
TE_FD_PLUS = ["te_fd_class_travel", "te_fd_cust_travel", "te_fd_dep", "te_fd_dep_arr", "te_fd_age", "te_persona"]


def add_te_fd_keys(m: Matrices) -> None:
    """Route keys (flight_distance, alone and x each categorical); TE fit per fold."""
    for frame in (m.X, m.X_test):
        fd = frame["flight_distance"].astype(str)
        frame["te_fd"] = fd
        frame["te_fd_class"] = fd + "|" + frame["class"].astype(str)
        frame["te_fd_travel"] = fd + "|" + frame["type_of_travel"].astype(str)
        frame["te_fd_cust"] = fd + "|" + frame["customer_type"].astype(str)


def add_te_fd_plus_keys(m: Matrices) -> None:
    for frame in (m.X, m.X_test):
        fd = frame["flight_distance"].astype(str)
        tr = frame["type_of_travel"].astype(str)
        frame["te_fd_class_travel"] = fd + "|" + frame["class"].astype(str) + "|" + tr
        frame["te_fd_cust_travel"] = fd + "|" + frame["customer_type"].astype(str) + "|" + tr


def add_flight_keys(m: Matrices, age: bool = False) -> None:
    for frame in (m.X, m.X_test):
        fd = frame["flight_distance"].astype(str)
        dep = frame["departure_delay_in_minutes"].astype(str)
        frame["te_fd_dep"] = fd + "|" + dep
        frame["te_fd_dep_arr"] = fd + "|" + dep + "|" + frame["arrival_delay_in_minutes"].astype(str)
        if age:
            frame["te_fd_age"] = fd + "|" + frame["age"].astype(str)


def add_persona_key(m: Matrices) -> None:
    for frame in (m.X, m.X_test):
        frame["te_persona"] = (frame["age"].astype(str) + "|" + frame["gender"].astype(str) + "|"
                               + frame["customer_type"].astype(str) + "|" + frame["type_of_travel"].astype(str))


def add_fd_age_key(m: Matrices) -> None:
    for frame in (m.X, m.X_test):
        frame["te_fd_age"] = frame["flight_distance"].astype(str) + "|" + frame["age"].astype(str)


def add_orig_fd_class(m: Matrices) -> None:
    """Original's smoothed rate per (flight_distance, class), shrunk to the fd rate."""
    orig = load_original().to_pandas()
    prior = float(orig[TARGET].mean())
    g = orig.groupby("flight_distance")[TARGET].agg(["sum", "count"])
    fd_rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
    gc = orig.groupby([orig["flight_distance"], orig["class"].astype(str)])[TARGET].agg(["sum", "count"])
    for frame in (m.X, m.X_test):
        base = frame["flight_distance"].map(fd_rate).astype(float).fillna(prior).to_numpy()
        idx = pd.MultiIndex.from_arrays([frame["flight_distance"], frame["class"].astype(str)])
        s_ = gc["sum"].reindex(idx).fillna(0).to_numpy()
        c_ = gc["count"].reindex(idx).fillna(0).to_numpy()
        frame["orig_fd_class_rate"] = (s_ + 10 * base) / (c_ + 10)


def add_route(m: Matrices, plus: bool = False) -> None:
    add_te_fd_keys(m)
    add_orig_fd(m)
    add_numcat(m, cols=["flight_distance"], cat=False)
    if plus:
        add_te_fd_plus_keys(m)
        add_orig_fd_class(m)


ORIG_KEYS = {
    "fdcats": ["flight_distance", "gender", "customer_type", "type_of_travel", "class"],
    "fdage": ["flight_distance", "age"],
    "fdr3": ["flight_distance", "online_boarding", "inflight_wifi_service", "seat_comfort"],
}


def add_orig_keys(m: Matrices) -> None:
    """Original's rate on finer route keys, shrunk toward the original fd rate."""
    orig = load_original().to_pandas()
    prior = float(orig[TARGET].mean())
    g = orig.groupby("flight_distance")[TARGET].agg(["sum", "count"])
    fd_rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
    for c in m.cat_cols:
        orig[c] = orig[c].astype(str)
    for frame in (m.X, m.X_test):
        base = frame["flight_distance"].map(fd_rate).astype(float).fillna(prior).to_numpy()
        for name, keys in ORIG_KEYS.items():
            gk = orig.groupby(keys)[TARGET].agg(["sum", "count"])
            idx = pd.MultiIndex.from_arrays(
                [frame[k].astype(str) if k in m.cat_cols else frame[k] for k in keys]
            )
            s_ = gk["sum"].reindex(idx).fillna(0).to_numpy()
            c_ = gk["count"].reindex(idx).fillna(0).to_numpy()
            frame[f"orig_{name}_rate"] = (s_ + 5 * base) / (c_ + 5)


def add_orig_resid(m: Matrices, k: float = 10.0) -> None:
    """Original's route residual: original y minus a route-free original-only model's
    5-fold OOF p, shrunk mean per fd. No synthetic labels (rule 2)."""
    from sklearn.model_selection import StratifiedKFold as SKF

    mo = build_matrices(with_original=True)
    assert mo.X_orig is not None and mo.y_orig is not None
    cols = [c for c in mo.feature_cols if c not in (ORIGINAL_FLAG, "flight_distance", "delay_per_mile")]
    xo, yo = mo.X_orig, mo.y_orig
    p = np.zeros(len(xo))
    for itr, iva in SKF(n_splits=5, shuffle=True, random_state=7).split(xo, yo):
        mdl = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1,
                             n_jobs=8, random_state=42).fit(xo.iloc[itr][cols], yo[itr])
        p[iva] = mdl.predict_proba(xo.iloc[iva][cols])[:, 1]
    g = pd.DataFrame({"fd": xo["flight_distance"].to_numpy(), "r": yo - p}).groupby("fd")["r"].agg(["sum", "count"])
    enc = g["sum"] / (g["count"] + k)
    for frame in (m.X, m.X_test):
        frame["orig_route_resid"] = enc.reindex(frame["flight_distance"].to_numpy()).fillna(0).to_numpy()


def add_route_agg(m: Matrices) -> None:
    """Route-level means of features over train+test (no labels), + row deviation."""
    both = pd.concat([m.X, m.X_test])
    cols = [*RATING_COLS, "age", "departure_delay_in_minutes"]
    num = both[cols].astype(float)
    num["is_business"] = (both["class"].astype(str) == "Business").astype(float)
    num["is_personal"] = (both["type_of_travel"].astype(str) == "Personal Travel").astype(float)
    num["is_loyal"] = (both["customer_type"].astype(str) == "Loyal Customer").astype(float)
    means = num.groupby(both["flight_distance"].to_numpy()).mean()
    for frame in (m.X, m.X_test):
        mm = means.loc[frame["flight_distance"].to_numpy()].to_numpy()
        for i, c in enumerate(means.columns):
            frame[f"fdmean_{c}"] = mm[:, i]


def add_orig_fd(m: Matrices) -> None:
    """Original's satisfaction rate / count per flight_distance (no synthetic labels)."""
    orig = load_original().to_pandas()
    prior = float(orig[TARGET].mean())
    g = orig.groupby("flight_distance")[TARGET].agg(["sum", "count"])
    rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
    for frame in (m.X, m.X_test):
        frame["orig_fd_rate"] = frame["flight_distance"].map(rate).astype(float).fillna(prior).to_numpy()
        frame["orig_fd_count"] = frame["flight_distance"].map(g["count"]).astype(float).fillna(0).to_numpy()


ARM_BUILDERS = {
    "profile": add_profile,
    "segment": add_segment,
    "numcat": add_numcat,
    "nc_cnt": lambda m: add_numcat(m, cat=False),
    "nc_cat": lambda m: add_numcat(m, cnt=False),
    "nc_fd": lambda m: add_numcat(m, cols=["flight_distance"]),
    "nc_age": lambda m: add_numcat(m, cols=["age"]),
    "nc_delays": lambda m: add_numcat(m, cols=NUMCAT_COLS[2:]),
    "te_fd": add_te_fd_keys,
    "orig_fd": add_orig_fd,
    "route": add_route,
    "route_plus": lambda m: add_route(m, plus=True),
    "route_origk": lambda m: (add_route(m), add_orig_keys(m)),
    "route_agg": lambda m: (add_route(m), add_route_agg(m)),
    "route_resid": add_route,
    "resid_only": lambda m: None,
    "route_resid_fine": add_route,
    "route_resid_ores": lambda m: (add_route(m), add_orig_resid(m)),
    "route_fdcat": lambda m: (add_route(m), add_numcat(m, cols=["flight_distance"], cnt=False)),
    "route_persona": lambda m: (add_route(m), add_persona_key(m)),
    "route_fdage2": lambda m: (add_route(m), add_fd_age_key(m)),
    "route_flight": lambda m: (add_route(m), add_flight_keys(m)),
    "route_fdage": lambda m: (add_route(m), add_flight_keys(m, age=True)),
}


RESID_KEYS: dict[str, list[str]] = {"route_resid": ["flight_distance"]}
RESID_KEYS_FINE: dict[str, list[str]] = {
    "route_resid": ["flight_distance"],
    "resid_fd_class": ["flight_distance", "class"],
    "resid_fd_travel": ["flight_distance", "type_of_travel"],
    "resid_fd_cust": ["flight_distance", "customer_type"],
}


def _key(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = frame[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + frame[c].astype(str)
    return out.to_numpy()


def route_residual(
    x_tr: pd.DataFrame, y_tr: np.ndarray, others: list[pd.DataFrame], k: float = 20.0,
    keys: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Fold-safe residual route encoding (PR-13).

    A route-free LGBM gives inner-OOF p for the training rows (5 inner folds) and a
    full-fit p for val/test. Per route: shrunk mean of (y - p) — the route's effect
    *conditional on* everything else. Training rows are encoded from the other
    inner folds only; val/test from the whole training split.
    """
    from sklearn.model_selection import StratifiedKFold as SKF

    base_cols = [c for c in x_tr.columns if not c.startswith(("te_fd", "TE_", "orig_fd", "cnt_flight", "fd_cat")) and not c.endswith("_cat")]
    params = {**PARAMS, "n_estimators": 200}
    p_tr = np.zeros(len(x_tr))
    resid_tr = np.zeros(len(x_tr))
    fd_tr = x_tr["flight_distance"].to_numpy()
    inner = list(SKF(n_splits=5, shuffle=True, random_state=7).split(x_tr, y_tr))
    for itr, iva in inner:
        mdl = LGBMClassifier(**params).fit(x_tr.iloc[itr][base_cols], y_tr[itr])
        p_tr[iva] = mdl.predict_proba(x_tr.iloc[iva][base_cols])[:, 1]
    r = y_tr - p_tr
    out_tr = x_tr.copy()
    outs_o = [f.copy() for f in others]
    for name, cols in (keys or RESID_KEYS).items():
        kt = _key(x_tr, cols)
        for itr, iva in inner:
            g = pd.DataFrame({"k": kt[itr], "r": r[itr]}).groupby("k")["r"].agg(["sum", "count"])
            resid_tr[iva] = (g["sum"] / (g["count"] + k)).reindex(kt[iva]).fillna(0).to_numpy()
        out_tr[name] = resid_tr.copy()
        g = pd.DataFrame({"k": kt, "r": r}).groupby("k")["r"].agg(["sum", "count"])
        enc_all = g["sum"] / (g["count"] + k)
        for f in outs_o:
            f[name] = enc_all.reindex(_key(f, cols)).fillna(0).to_numpy()
    return (out_tr, *outs_o)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "orig", "te_pairs", "cat_ratings", *ARM_BUILDERS])
    args = parser.parse_args()

    m = build_matrices(with_original=args.arm == "orig")
    m.X, m.X_test = m.X.copy(), m.X_test.copy()
    if args.arm == "orig":
        add_orig_features(m)
    elif args.arm == "te_pairs":
        add_pair_keys(m)
    elif args.arm == "cat_ratings":
        for frame in (m.X, m.X_test):
            for col in RATING_COLS:
                frame[col] = pd.Categorical(frame[col], categories=list(range(6)))
    elif args.arm in ARM_BUILDERS:
        ARM_BUILDERS[args.arm](m)
    print(f"arm {args.arm}: train {m.X.shape}  test {m.X_test.shape}", flush=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        te_cols = [c for c in (*TE_FD_COLS, *TE_FD_PLUS) if c in x_tr.columns]
        if te_cols:
            x_tr, x_va, x_te = add_fold_safe_target_encoding(
                x_tr, y_tr, [x_va, x_te], te_cols, {1: "sat"},
                smooth=TE_SMOOTH,
            )
            x_tr, x_va, x_te = (f.drop(columns=te_cols) for f in (x_tr, x_va, x_te))
        if args.arm in ("route_resid", "resid_only", "route_resid_ores"):
            x_tr, x_va, x_te = route_residual(x_tr, y_tr, [x_va, x_te], k=float(os.environ.get("S6E10_RESID_K", "20")))
        if args.arm == "route_resid_fine":
            x_tr, x_va, x_te = route_residual(x_tr, y_tr, [x_va, x_te], keys=RESID_KEYS_FINE)
        if args.arm == "te_pairs":
            x_tr, x_va, x_te = add_fold_safe_target_encoding(
                x_tr, y_tr, [x_va, x_te], PAIR_COLS, {1: "sat"}
            )
            # The raw integer keys stay out of the model: TE is the feature.
            x_tr, x_va, x_te = (f.drop(columns=PAIR_COLS) for f in (x_tr, x_va, x_te))
        import json as _json
        model = LGBMClassifier(**{**PARAMS, **_json.loads(os.environ.get("S6E10_LGBM_EXTRA", "{}"))})
        model.fit(
            x_tr, y_tr, eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        print(
            f"  fold {fold}: best_iter={model.best_iteration_} "
            f"AUC {roc_auc_score(y_va, model.predict_proba(x_va)[:, 1]):.5f} "
            f"({time.time() - t0:.0f}s, {x_tr.shape[1]} feats)",
            flush=True,
        )
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(m.X, m.y, m.X_test, fit_fold, n_outputs=1, score_fn=fold_score)
    oof_auc = float(roc_auc_score(m.y, oof))
    name = f"lgbm_arm_{args.arm}" + os.environ.get("S6E10_ARM_SUFFIX", "")
    save_cv_result(RESULTS_DIR, name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{name}.npy", test)

    # Paired read against the PR-1 control, per fold on the same splits.
    from sklearn.model_selection import StratifiedKFold

    ctrl = np.load(RESULTS_DIR / f"oof_{CONTROL}.npy")
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
    deltas = [
        roc_auc_score(m.y[va], oof[va]) - roc_auc_score(m.y[va], ctrl[va]) for _, va in splits
    ]
    d_oof = oof_auc - float(roc_auc_score(m.y, ctrl))
    n_pos = sum(d > 0 for d in deltas)
    verdict = "PASS" if d_oof >= BAR and n_pos >= 4 else "FAIL"
    print(f"\n{name}: OOF {oof_auc:.6f}  per-fold delta {np.round(deltas, 6).tolist()}")
    print(f"{name}: OOF delta {d_oof:+.6f}  positive folds {n_pos}/5  → {verdict}")
    print(f"corr with control {np.corrcoef(oof, ctrl)[0, 1]:.4f}")
    route_path = RESULTS_DIR / f"oof_{ROUTE_CONTROL}.npy"
    if name != ROUTE_CONTROL and route_path.exists():
        rc = np.load(route_path)
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
        rd = [roc_auc_score(m.y[va], oof[va]) - roc_auc_score(m.y[va], rc[va]) for _, va in splits]
        print(f"vs {ROUTE_CONTROL}: OOF delta {oof_auc - float(roc_auc_score(m.y, rc)):+.6f}  "
              f"per-fold {np.round(rd, 6).tolist()}")


if __name__ == "__main__":
    main()
