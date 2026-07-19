"""ORIGINAL-50K REFEREE (pre-registered Day-15 queue, user-approved 07-17).

Question: is any finals candidate differentially overfit to generator
flip-noise? data/external/student_health_dataset_50k.csv (the generator's
source: exact 13-feature schema, zero missing, labels rule-consistent 0/50k)
is the one labeled set whose noise is INDEPENDENT of our 690k CV rows. It has
only ever been used as augmentation (vetoed Day-3) and linkage (dead Day-4) —
never as a VALIDATION instrument.

Design (CLAUDE.md Day-15 queue):
1. Schema/level alignment check original <-> features.py.
2. Seed-42 legs retrained with the referee frame as a SECOND test frame
   through the exact test-side pipeline (per-fold TE encoders, 5-fold test
   averaging): lgbm / xgboost / catboost (champion-era configs, water kept),
   hgbc_te3, lgbm_te3, realmlp_te13.
3. Referee-compose the finals candidates from those leg predictions with the
   DEPLOYED recipes read from disk: champion_v1 (per-model precorrect on _v1
   OOF + NM blend + decision weights — forensics_v1b_v2 recipe), the u-stack
   (stack_oof LR meta over the deployed OOF arrays with the referee frame in
   the test slot + robust decision weights), realmlp_te13 solo (saved breadth
   decision weights).
4. Balanced accuracy on original labels. PRE-REGISTERED READ: RANKING + GAP
   PATTERN ONLY, never absolutes (distribution shift: no missingness, no
   flips, contracted tails). All candidates in a common band = NULL (expected).
   A candidate materially below its peers = flip-noise-overfit flag -> demote
   on the finals #2 menu. Blind spot: no NaNs here, so this instrument says
   NOTHING about missing-driver behavior (26% of competition rows).

Known approximations (recorded up front, all rank-preserving under the
pre-registered read): seed-42 single-seed legs stand in for the 3-seed bagw /
8-seed breadth deployed legs, and the u-stack referee composition drops the
mlp + hgbc_bagw seats (6 of the 8 KEYS8 legs) because those trainers are not
part of the pre-registered leg list — the meta is refit on the matching 6-leg
OOF subset so train- and referee-side features agree.

Run:  python src/probe_original_referee.py          (~1-2h; GPU legs last)
Resumable per (leg, fold) via results/cache/refprobe_*/ checkpoints.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES, NUM_COLS, TARGET
from lb_anchor import correct
from train_common import (
    DATA_DIR,
    N_CLASSES,
    N_FOLDS,
    RESULTS_DIR,
    Dataset,
    load_dataset,
    robust_decision_weights,
)
from train_fe import ORDINAL, combo_key
from train_te_num import encode_fold, with_te
from zoo_common import CACHE_DIR, te13_block_for_fold, te_block_for_fold

from data_science_stuff.kaggle.stacking import stack_oof
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"
REPORT = RESULTS_DIR / "probe_original_referee.txt"
STACK_KEYS6 = [
    "lgbm_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
    "realmlp_te13_u_breadth",
]
# which regenerated referee leg stands in for each deployed stack key
STACK_LEG_FOR_KEY = {
    "lgbm_bagw": "lgbm",
    "xgboost_bagw": "xgboost",
    "catboost_bagw": "catboost",
    "hgbc_te3_u_breadth": "hgbc_te3",
    "lgbm_te3_u_breadth": "lgbm_te3",
    "realmlp_te13_u_breadth": "realmlp_te13",
}

_report_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line, flush=True)
    _report_lines.append(line)


# ---------------------------------------------------------------- referee frame


def load_referee(ds: Dataset) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Original 50k as a second test frame, dtypes matched to ds.train."""
    ext = pl.read_csv(EXT_CSV).to_pandas()
    assert len(ext) == 50_000, len(ext)
    for c in ds.feature_cols:
        assert c in ext.columns, f"external missing feature {c}"
    assert set(ext[TARGET].unique()) == set(CLASSES), ext[TARGET].unique()
    y_ref = ext[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X_ref = ext[ds.feature_cols].copy()
    # dtype match matters: string-cast TE views ("7000" vs "7000.0") and
    # exact-value te_maps lookups both key on the value's repr/dtype
    for c in X_ref.columns:
        X_ref[c] = X_ref[c].astype(ds.train[c].dtype)
    return X_ref, y_ref.astype(np.int64)


def schema_check(ds: Dataset, X_ref: pd.DataFrame, y_ref: NDArray[np.int64]) -> None:
    emit("== schema/level alignment: original 50k vs competition frames ==")
    assert int(X_ref.isna().sum().sum()) == 0, "original must be NaN-free"
    emit("NaN in original: 0 (asserted)")
    for c in ds.cat_cols:
        ref_v = set(X_ref[c].unique())
        tr_v = set(ds.train[c].dropna().unique())
        assert ref_v <= tr_v, f"{c}: original levels {ref_v - tr_v} unseen in train"
        emit(f"  cat {c}: {len(ref_v)} levels, subset of train vocab OK")
    for c in [n for n in NUM_COLS if n in X_ref.columns] + ["water_intake"]:
        lo, hi = float(X_ref[c].min()), float(X_ref[c].max())
        tlo, thi = float(ds.train[c].min()), float(ds.train[c].max())
        tag = "OK" if lo >= tlo and hi <= thi else "WIDER-THAN-TRAIN"
        emit(f"  num {c}: ref [{lo:g}, {hi:g}] vs train [{tlo:g}, {thi:g}] {tag}")
    n_ref_sleep = X_ref["sleep_duration"].nunique()
    n_tr_sleep = ds.train["sleep_duration"].nunique()
    emit(f"  sleep_duration uniques: ref {n_ref_sleep} vs train {n_tr_sleep} "
         "(700 vs 701 expected)")
    prior_ref = np.bincount(y_ref, minlength=N_CLASSES) / len(y_ref)
    prior_tr = np.bincount(ds.y, minlength=N_CLASSES) / len(ds.y)
    emit("  class priors ref  " + " ".join(f"{c}={p:.4f}" for c, p in zip(CLASSES, prior_ref)))
    emit("  class priors train" + " ".join(f" {c}={p:.4f}" for c, p in zip(CLASSES, prior_tr)))
    emit()


# ------------------------------------------------------------------- leg runner

# fit_fold(tr_idx, val_idx, fold) -> (val_proba, ref_proba)
RefFitFold = Callable[
    [NDArray[np.int64], NDArray[np.int64], int],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


def referee_cv(
    ds: Dataset, n_ref: int, fit_fold: RefFitFold, leg: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """zoo_cv twin: identical seed-42 splits, referee frame in the test slot,
    per-fold checkpoints (this box hard-reboots at random)."""
    ckpt = CACHE_DIR / f"refprobe_{leg}"
    ckpt.mkdir(parents=True, exist_ok=True)
    oof = np.zeros((len(ds.y), N_CLASSES))
    ref = np.zeros((n_ref, N_CLASSES))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        vck, rck = ckpt / f"f{fold}_val.npy", ckpt / f"f{fold}_ref.npy"
        if vck.exists() and rck.exists():
            val_proba, ref_proba = np.load(vck), np.load(rck)
            print(f"  [{leg}] fold {fold}: loaded checkpoint", flush=True)
        else:
            val_proba, ref_proba = fit_fold(tr_idx, val_idx, fold)
            np.save(vck, val_proba)
            np.save(rck, ref_proba)
        oof[val_idx] = val_proba
        ref += ref_proba / N_FOLDS
        score = balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
        print(f"  [{leg}] fold {fold} val balanced_acc (argmax): {score:.4f}", flush=True)
    return oof, ref


# ------------------------------------------------------------------------- legs


def leg_lgbm(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    from baseline import load_params

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    R = X_ref.copy()
    for col in ds.cat_cols:
        R[col] = pd.Categorical(R[col], categories=X[col].cat.categories)
    params = load_params()

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        model = LGBMClassifier(**{**params, "random_state": SEED})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(R)

    return referee_cv(ds, len(X_ref), fit_fold, "lgbm")


def leg_xgboost(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    from sklearn.utils.class_weight import compute_sample_weight

    from train_xgboost import XGBClassifier, load_params

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    R = X_ref.copy()
    for col in ds.cat_cols:
        R[col] = pd.Categorical(R[col], categories=X[col].cat.categories)
    params = load_params()

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        sw = compute_sample_weight("balanced", ds.y[tr_idx])
        model = XGBClassifier(**{**params, "random_state": SEED})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            sample_weight=sw,
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            verbose=False,
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        out = model.predict_proba(X.iloc[val_idx]), model.predict_proba(R)
        del model
        return out

    return referee_cv(ds, len(X_ref), fit_fold, "xgboost")


def leg_catboost(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    from catboost import CatBoostClassifier, Pool

    from train_catboost import CAT_FILL, load_params

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna(CAT_FILL).astype(str)
    R = X_ref.copy()
    for col in ds.cat_cols:
        R[col] = R[col].astype(str)
    params = load_params()
    print(f"  [catboost] task_type: {params['task_type']}", flush=True)
    cat_idx = [X.columns.get_loc(c) for c in ds.cat_cols]
    ref_pool = Pool(R, cat_features=cat_idx)

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        tr_pool = Pool(X.iloc[tr_idx], ds.y[tr_idx], cat_features=cat_idx)
        val_pool = Pool(X.iloc[val_idx], cat_features=cat_idx)
        model = CatBoostClassifier(**{**params, "random_state": SEED})
        model.fit(tr_pool)  # no eval_set — 6 GB GPU CTR rule (train_catboost)
        assert [int(c) for c in model.classes_] == list(range(N_CLASSES))
        out = model.predict_proba(val_pool), model.predict_proba(ref_pool)
        del model, tr_pool, val_pool
        return out

    return referee_cv(ds, len(X_ref), fit_fold, "catboost")


def leg_hgbc_te3(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from train_hgbc_te3 import HGBC_CONFIG

    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    R = X_ref.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        R[col] = R[col].astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
        R[col] = pd.Categorical(R[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_ref_view = R[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
        )

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_rf = enc.transform(te_ref_view)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
        model.fit(assemble(X.iloc[tr_idx], te_tr), ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return (
            model.predict_proba(assemble(X.iloc[val_idx], te_va)),
            model.predict_proba(assemble(R, te_rf)),
        )

    return referee_cv(ds, len(X_ref), fit_fold, "hgbc_te3")


def leg_lgbm_te3(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    from train_lgbm_te3 import LGBM_TE3_PARAMS

    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    R = X_ref.copy()
    for col in ds.cat_cols:
        R[col] = pd.Categorical(R[col], categories=X[col].cat.categories)
    te_tr_view = ds.train[raw_cols].astype(str)
    te_ref_view = X_ref[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
        )

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_rf = enc.transform(te_ref_view)
        model = LGBMClassifier(**{**LGBM_TE3_PARAMS, "random_state": SEED})
        model.fit(
            assemble(X.iloc[tr_idx], te_tr),
            ds.y[tr_idx],
            eval_set=[(assemble(X.iloc[val_idx], te_va), ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return (
            model.predict_proba(assemble(X.iloc[val_idx], te_va)),
            model.predict_proba(assemble(R, te_rf)),
        )

    return referee_cv(ds, len(X_ref), fit_fold, "lgbm_te3")


def leg_realmlp_te13(ds: Dataset, X_ref: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """train_realmlp_td fit_fold twin (TE13=1, u surface) + referee inputs."""
    import torch

    from train_mlp import NNData

    from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

    CFG = dict(
        train_bs=512,
        fused_optimizer=True,
        allow_amp=True,
        verbosity=1,
        epoch_select="bacc",
        snapshot_k=0,
    )
    nd = NNData(ds)
    # referee-side NNData twin: same vocab maps (train|test), no NaN -> zero masks
    cat_ref_cols = []
    for c in nd.cat_cols:
        cats = sorted(set(ds.train[c].dropna()) | set(ds.test[c].dropna()))
        m = {v: i + 1 for i, v in enumerate(cats)}
        cat_ref_cols.append(X_ref[c].map(m).fillna(0).to_numpy(dtype=np.int64))
    Xcat_ref = np.stack(cat_ref_cols, axis=1)
    Xnum_ref = X_ref[nd.num_cols].to_numpy(dtype=np.float64)
    miss_ref = np.zeros((len(X_ref), nd.miss_tr.shape[1]))  # NaN-free frame

    cat_names = [f"cat_{c}" for c in nd.cat_cols]
    miss_names = [f"miss_{i}" for i in range(nd.miss_tr.shape[1])]

    def frame(cat, num_filled, miss, te) -> pd.DataFrame:  # noqa: ANN001
        blocks = {
            **{n: cat[:, i] for i, n in enumerate(cat_names)},
            **{n: num_filled[:, i] for i, n in enumerate(nd.num_cols)},
            **{n: miss[:, i] for i, n in enumerate(miss_names)},
            **{f"te_{i}": te[:, i] for i in range(te.shape[1])},
        }
        return pd.DataFrame(blocks)

    # referee TE-block inputs (te_block_for_fold twin, transformed per fold below)
    num_cols = [c for c in NUM_COLS if c in ds.train.columns]
    src_tr = ds.train[num_cols].copy()
    src_tr["_combo"] = combo_key(ds.train)
    src_ref = X_ref[num_cols].copy()
    src_ref["_combo"] = combo_key(X_ref)
    te_cols = [*num_cols, "_combo"]
    cols13 = list(ds.train.columns)
    tr13_view = ds.train[cols13].astype(str)
    ref13_view = X_ref[cols13].astype(str)

    def ordinal_block(df: pd.DataFrame) -> NDArray[np.float32]:
        out = np.full((len(df), len(ORDINAL)), np.nan, dtype=np.float32)
        for i, (col, m) in enumerate(ORDINAL.items()):
            out[:, i] = df[col].map(m).to_numpy(dtype=np.float32)
        return out

    ord_ref = ordinal_block(X_ref)

    def fit_fold(tr_idx, val_idx, fold):  # noqa: ANN001, ANN202
        fold_seed = SEED + fold * 100
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)  # noqa: NPY002 — library-internal RNG expects it

        # train/test blocks from the shared cache (identical to the deployed run)
        te_tr_full, _te_te, _names = te_block_for_fold(ds, tr_idx, SEED, fold, "u")
        t13_tr, _t13_te, _n13 = te13_block_for_fold(ds, tr_idx, SEED, fold, "u")
        te_tr_full = np.concatenate([te_tr_full, t13_tr], axis=1)

        # referee transforms from freshly refit fold encoders (cache holds no
        # encoder objects; encode_fold/TargetEncoder are deterministic per fold)
        _te_tr, full_maps, prior = encode_fold(src_tr, ds.y, tr_idx, te_cols, SEED)
        ref_te = with_te(src_ref, full_maps, prior)
        te_names = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASSES)]
        ref_block = ref_te[te_names].to_numpy(dtype=np.float32)
        prior_row = np.tile(prior.astype(np.float32), len(te_cols))
        nan_r, nan_c = np.where(np.isnan(ref_block))
        ref_block[nan_r, nan_c] = prior_row[nan_c]
        enc13 = TargetEncoder(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            smooth="auto",
            target_type="multiclass",
        )
        enc13.fit(tr13_view.iloc[tr_idx], ds.y[tr_idx])
        ref13 = enc13.transform(ref13_view).astype(np.float32)
        te_ref_full = np.concatenate([ref_block, ord_ref, ref13], axis=1)
        assert te_ref_full.shape[1] == te_tr_full.shape[1], (
            te_ref_full.shape, te_tr_full.shape,
        )

        num_med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
        num_tr = np.where(np.isnan(nd.Xnum_tr), num_med, nd.Xnum_tr)
        num_rf = np.where(np.isnan(Xnum_ref), num_med, Xnum_ref)
        te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
        te_tr_f = np.where(np.isnan(te_tr_full), te_med, te_tr_full)
        te_rf_f = np.where(np.isnan(te_ref_full), te_med, te_ref_full)

        X_all = frame(nd.Xcat_tr, num_tr, nd.miss_tr, te_tr_f)
        X_rf = frame(Xcat_ref, num_rf, miss_ref, te_rf_f)
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]

        model = RealMLP_TD_Classifier(**{**CFG, "random_state": fold_seed})
        model.fit(
            X_tr,
            ds.y[tr_idx],
            X_val,
            ds.y[val_idx],
            cat_col_names=cat_names,
            X_test=X_rf,
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        val_proba = model.best_val_probs_.astype(np.float64)
        ref_proba = model.predict_proba(X_rf).astype(np.float64)
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        ref_proba /= ref_proba.sum(axis=1, keepdims=True)

        del model, X_all, X_rf, X_tr, X_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_proba, ref_proba

    return referee_cv(ds, len(X_ref), fit_fold, "realmlp_te13")


# ------------------------------------------------------------------- candidates


def champion_v1_ref(
    y: NDArray[np.int64], ref_legs: dict[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """forensics_v1b_v2.champion_test_labels with the referee frame in the test
    slot: per-model precorrect + NM blend + decision weights, all fit on the
    deployed _v1 OOF arrays, applied to the referee leg predictions."""
    from scipy.optimize import minimize

    gbdts = ["lgbm", "xgboost", "catboost"]
    oof = {m: np.load(RESULTS_DIR / f"oof_{m}_v1.npy") for m in gbdts}
    w = {m: tune_decision_weights(y, oof[m]) for m in gbdts}
    pc_oof = [correct(oof[m], w[m]) for m in gbdts]
    pc_ref = [correct(ref_legs[m], w[m]) for m in gbdts]

    def blend(wv: NDArray, arrays: list[NDArray]) -> NDArray:
        s = np.exp(wv) / np.exp(wv).sum()
        return sum(s[i] * a for i, a in enumerate(arrays))

    res = minimize(
        lambda wv: (
            -float(balanced_accuracy_score(y, np.argmax(blend(wv, pc_oof), axis=1)))
        ),
        np.ones(len(gbdts)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    blended_oof = blend(res.x, pc_oof)
    dw = tune_decision_weights(y, blended_oof)
    oof_score = balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    emit(f"champion_v1 recipe refit: OOF weighted {oof_score:.5f} "
         "(recorded reconstruction 0.9494-era)")
    return correct(blend(res.x, pc_ref), dw)  # normalized weighted proba


def ustack_ref(
    y: NDArray[np.int64], ref_legs: dict[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Deployed u-stack procedure (stack_oof LR meta + robust decision weights)
    over the 6-leg OOF subset, referee frame in the test slot."""
    from sklearn.linear_model import LogisticRegression

    s_oof_ck = CACHE_DIR / "refprobe_ustack_soof.npy"
    s_ref_ck = CACHE_DIR / "refprobe_ustack_sref.npy"
    if s_oof_ck.exists() and s_ref_ck.exists():
        s_oof, s_ref = np.load(s_oof_ck), np.load(s_ref_ck)
        dw = robust_decision_weights(y, s_oof)
        oof_score = balanced_accuracy_score(y, weighted_predict(s_oof, dw))
        emit(f"u-stack (6-leg subset, cached refit): OOF weighted {oof_score:.5f} "
             "(deployed 8-leg u4 reads 0.95081; subset drops mlp+hgbc_bagw)")
        return correct(s_ref, dw)

    base_oofs = [np.load(RESULTS_DIR / f"oof_{k}.npy") for k in STACK_KEYS6]
    base_refs = [ref_legs[STACK_LEG_FOR_KEY[k]] for k in STACK_KEYS6]

    def meta_factory() -> LogisticRegression:
        return LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, n_jobs=-1
        )

    s_oof, s_ref = stack_oof(base_oofs, base_refs, y, meta_factory, use_logits=True)
    np.save(s_oof_ck, s_oof)
    np.save(s_ref_ck, s_ref)
    dw = robust_decision_weights(y, s_oof)
    oof_score = balanced_accuracy_score(y, weighted_predict(s_oof, dw))
    emit(f"u-stack (6-leg subset) refit: OOF weighted {oof_score:.5f} "
         "(deployed 8-leg u4 reads 0.95081; subset drops mlp+hgbc_bagw)")
    return correct(s_ref, dw)


def main() -> None:
    import os

    for env in ("S6E7_REPAIR", "S6E7_REPAIR_FIT", "S6E7_SEEDS", "S6E7_N_FOLDS"):
        assert not os.environ.get(env), f"{env} must be unset (u surface, K=5, s42)"
    ds = load_dataset()
    X_ref, y_ref = load_referee(ds)
    emit("ORIGINAL-50K REFEREE — pre-registered read: RANKING + GAP PATTERN ONLY")
    emit("(absolutes are shifted by construction: no missingness, no label flips)")
    emit()
    schema_check(ds, X_ref, y_ref)

    legs: dict[str, NDArray[np.float64]] = {}
    leg_oofs: dict[str, NDArray[np.float64]] = {}
    runners = {
        "hgbc_te3": leg_hgbc_te3,
        "lgbm_te3": leg_lgbm_te3,
        "lgbm": leg_lgbm,
        "xgboost": leg_xgboost,
        "catboost": leg_catboost,
        "realmlp_te13": leg_realmlp_te13,  # GPU, longest — last
    }
    emit("== seed-42 leg regeneration (referee frame in the test slot) ==")
    for leg, runner in runners.items():
        print(f"\n[leg {leg}]", flush=True)
        oof, ref = runner(ds, X_ref)
        legs[leg] = ref
        leg_oofs[leg] = oof
        oof_ba = balanced_accuracy_score(ds.y, oof.argmax(axis=1))
        ref_ba = balanced_accuracy_score(y_ref, ref.argmax(axis=1))
        emit(f"  leg {leg:14s} OOF argmax {oof_ba:.5f}   referee argmax {ref_ba:.5f}")
    emit()

    emit("== referee-composed finals candidates ==")
    cands: dict[str, NDArray[np.float64]] = {}
    cands["champion_v1"] = champion_v1_ref(ds.y, legs)
    cands["lrstack_u4(6-leg)"] = ustack_ref(ds.y, legs)
    with (RESULTS_DIR / "decision_weights_realmlp_te13_u_breadth.json").open() as f:
        rw_map = json.load(f)
    rw = np.array([rw_map[c] for c in CLASSES])
    cands["realmlp_te13 solo"] = correct(legs["realmlp_te13"], rw)
    emit()

    emit("== REFEREE BOARD (balanced accuracy on original 50k labels) ==")
    rows = []
    for name, proba in cands.items():
        pred = proba.argmax(axis=1)
        ba = balanced_accuracy_score(y_ref, pred)
        rec = [
            float((pred[y_ref == k] == k).mean()) for k in range(N_CLASSES)
        ]
        rows.append((name, ba, rec))
    rows.sort(key=lambda r: -r[1])
    top = rows[0][1]
    for name, ba, rec in rows:
        recs = " ".join(f"{c}={r:.4f}" for c, r in zip(CLASSES, rec))
        emit(f"  {name:20s} BA {ba:.5f}  gap-to-top {ba - top:+.5f}   {recs}")
    emit()
    emit("read: all candidates within a common band = NULL (expected outcome).")
    emit("a candidate materially below its peers = flip-noise-overfit flag.")
    emit("blind spot: instrument referees the complete-driver regime only.")

    REPORT.write_text("\n".join(_report_lines) + "\n")
    print(f"\nreport -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
