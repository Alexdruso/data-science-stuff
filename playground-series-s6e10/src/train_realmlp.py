"""RealMLP-TD leg for PS S6E10 — a model that differs in kind from the GBDTs.

    python src/train_realmlp.py               # 5-fold, saves oof/test_realmlp_v1.npy
    python src/train_realmlp.py --epochs 24
    python src/train_realmlp.py --fold1 --tag e8 --override '{"epochs": 8}'

Wiring follows s6e8's `train_realmlp.py`, including its two package traps:

1. `NumericalPreprocessor` does not impute (a NaN median poisons the column), so
   numeric NaNs are filled with **fold-train medians** here.
2. `fit()` casts categoricals with `.astype(np.int64)` and clips to the observed
   max, so they must arrive as integer codes; +1 gives a pandas -1 (NaN) its own
   level instead of merging it into level 0.

`epoch_select="auc"` matters: on s6e8 RealMLP peaked mid-run and then overfit
monotonically. Note the selected epoch is chosen on the val fold it is scored on,
so solo OOF is slightly optimistic (see CLAUDE.md PR-2).

GPU memory rule: `fit` frees nothing, so the model is deleted and the allocator
cleared inside every fold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET
from matrices import build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.models import REALMLP_TD_CONFIG, RealMLP_TD_Classifier

_, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
CKPT_DIR = RESULTS_DIR / "realmlp_ckpt"

# Overrides on top of REALMLP_TD_CONFIG (s6e8-proven for binary AUC).
CFG: dict[str, object] = {
    "epoch_select": "auc",
    # The balanced-softmax prior is an imbalance hack; 44% positive needs none.
    "loss_prior_power": 0.0,
    "train_bs": 512,
    "allow_amp": True,
    "fused_optimizer": True,
    "epochs": 16,
    "verbosity": 1,
}


def to_frames(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], cat_cols: list[str]
) -> list[pd.DataFrame]:
    """NaN-free frames, categoricals as non-negative integer codes.

    Medians come from the fold's training split only (fold-aware statistics).
    """
    num_cols = [c for c in frames[0].columns if c not in cat_cols]
    medians = frames[0][num_cols].median().fillna(0.0)
    out: list[pd.DataFrame] = []
    for frame in frames:
        block = frame[num_cols].astype(np.float64).fillna(medians)
        for col in cat_cols:
            block[col] = frame[col].cat.codes.to_numpy() + 1
        out.append(block)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", default="{}", help="JSON dict of config overrides")
    parser.add_argument("--route", action="store_true", help="PR-12 route features")
    parser.add_argument("--fdcat", action="store_true", help="flight_distance as an embedded category")
    parser.add_argument("--fdmin", type=int, default=20, help="min count for an fd embedding level")
    parser.add_argument("--delaycat", action="store_true", help="delays (count>=20 levels) as embedded categories")
    parser.add_argument("--fdclasscat", action="store_true", help="(fd, class) as one embedded category")
    parser.add_argument("--orig", action="store_true", help="append original rows to each training fold")
    parser.add_argument("--origrep", type=int, default=1, help="replicate the original rows N times (with --orig)")
    parser.add_argument("--orignodelay", action="store_true", help="original rows' delay columns → NaN (fold-median filled)")
    parser.add_argument("--nfolds", type=int, default=5, help="CV folds (PR-18: 10 = models on 90%% of train)")
    parser.add_argument("--extras", default="", help="comma list of og,agg,aux,auxall (route.add_extras)")
    parser.add_argument("--pseudo", type=float, default=0.0,
                        help="PR-25: >0 = refit on train + test rows with labels SAMPLED from this fold's own model (weight via repeats ignored)")
    parser.add_argument("--soft", type=float, default=0.0,
                        help="PR-30 soft labels: target = (1-a)·y + a·p, p = inner 3-fold LGBM on the "
                             "fold's synthetic training rows (exact soft CE via weighted row duplication)")
    parser.add_argument("--soft-teacher", default="",
                        help="with --soft: p = this OOF array (results/oof_<name>.npy, id order) instead of inner LGBM")
    parser.add_argument("--soft-inner-nn", action="store_true",
                        help="with --soft: p = inner 3-fold RealMLP (honest born-again teacher)")
    parser.add_argument("--agecat", action="store_true", help="age as an embedded category")
    parser.add_argument("--ratingcat", action="store_true", help="the 13 ratings as embedded categories")
    parser.add_argument("--resid", action="store_true", help="PR-13 residual route encoding (with --route)")
    parser.add_argument("--tag", default="", help="run-name suffix; required with --override")
    parser.add_argument(
        "--fullfit", action="store_true",
        help="fit once on ALL train rows (fixed epochs; needs epoch_select=last), save test_<run>_full.npy",
    )
    parser.add_argument(
        "--fold1", action="store_true",
        help="fit fold 1 only (same split as run_cv) and print its AUC; saves no arrays",
    )
    args = parser.parse_args()

    overrides = json.loads(args.override)
    if overrides and not args.tag:
        raise SystemExit("--override needs --tag, or the arrays overwrite realmlp_v1")
    cfg = {**REALMLP_TD_CONFIG, **CFG, **overrides}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    m = build_matrices(with_original=args.orig)
    n_test = len(m.X_test)
    if args.orig:
        # Ride along with X_test through every global feature builder, split back below.
        m.X_test = pd.concat([m.X_test, m.X_orig], ignore_index=True)
    if args.route:
        from route import add_route_global
        add_route_global(m)
        args.tag = args.tag or "route"
    cat_cols = list(m.cat_cols)
    if args.fdcat:
        both = pd.concat([m.X, m.X_test])
        counts = both["flight_distance"].value_counts()
        levels = counts.index[counts >= args.fdmin]
        for frame in (m.X, m.X_test):
            frame["fd_cat"] = pd.Categorical(
                frame["flight_distance"].where(frame["flight_distance"].isin(levels)), categories=levels
            )
        cat_cols.append("fd_cat")
        args.tag += "_fdcat" + ("" if args.fdmin == 20 else f"{args.fdmin}")
    if args.agecat:
        for frame in (m.X, m.X_test):
            frame["age_cat"] = pd.Categorical(frame["age"], categories=sorted(pd.concat([m.X, m.X_test])["age"].unique()))
        cat_cols.append("age_cat")
        args.tag += "_agecat"
    if args.fdclasscat:
        key = lambda f: f["flight_distance"].astype(str) + "|" + f["class"].astype(str)
        counts = pd.concat([key(m.X), key(m.X_test)]).value_counts()
        levels = counts.index[counts >= 20]
        for frame in (m.X, m.X_test):
            k = key(frame)
            frame["fdclass_cat"] = pd.Categorical(k.where(k.isin(levels)), categories=levels)
        cat_cols.append("fdclass_cat")
        args.tag += "_fdclasscat"
    if args.delaycat:
        both = pd.concat([m.X, m.X_test])
        for col in ("departure_delay_in_minutes", "arrival_delay_in_minutes"):
            counts = both[col].value_counts()
            levels = counts.index[counts >= 20]
            for frame in (m.X, m.X_test):
                frame[f"{col}_cat"] = pd.Categorical(frame[col].where(frame[col].isin(levels)), categories=levels)
            cat_cols.append(f"{col}_cat")
        args.tag += "_delaycat"
    if args.ratingcat:
        from features import RATING_COLS
        for frame in (m.X, m.X_test):
            for col in RATING_COLS:
                frame[f"{col}_cat"] = pd.Categorical(frame[col], categories=list(range(6)))
        cat_cols += [f"{col}_cat" for col in RATING_COLS]
        args.tag += "_ratingcat"
    if args.resid:
        args.tag += "_resid"
    x_orig = None
    if args.orig:
        x_orig = m.X_test.iloc[n_test:].reset_index(drop=True)
        m.X_test = m.X_test.iloc[:n_test].reset_index(drop=True)
        args.tag += "_orig"
        if args.orignodelay:
            # Original delays are ~12x the synthetic ones (off-distribution).
            for col in x_orig.columns:
                if "delay" in col and not col.endswith("_cat") and x_orig[col].dtype.kind == "f" or col in (
                    "departure_delay_in_minutes", "arrival_delay_in_minutes"
                ):
                    x_orig[col] = np.nan
            args.tag += "nd"
        if args.origrep > 1:
            args.tag += f"x{args.origrep}"
    extras = [e for e in args.extras.split(",") if e]
    if extras:
        from route import add_extras
        add_extras(m, x_orig, extras)
        args.tag += "".join(f"_{e}" for e in extras)
    # is_original is constant 0 here; RealMLP's robust scaler has nothing to scale.
    drop = [] if args.orig else ["is_original"]
    X = m.X.drop(columns=drop)
    X_test = m.X_test.drop(columns=drop)
    print(
        f"RealMLP on {'cuda' if torch.cuda.is_available() else 'cpu'}: "
        f"train {X.shape} test {X_test.shape} epochs={cfg['epochs']} "
        f"n_ens={cfg['n_ens']} train_bs={cfg['train_bs']}",
        flush=True,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        row_ids = x_tr.index.to_numpy()
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        if args.route:
            from route import route_te
            if x_orig is not None:
                x_tr, x_va, x_te, x_o = route_te(x_tr, y_tr, [x_va, x_te, x_orig])
            else:
                x_tr, x_va, x_te = route_te(x_tr, y_tr, [x_va, x_te])
        n_syn = len(x_tr)
        soft_t = None
        if args.soft > 0 and args.soft_teacher:
            tch = np.load(RESULTS_DIR / f"oof_{args.soft_teacher}.npy")[row_ids]
            p_in = 1 / (1 + np.exp(-tch)) if (tch.min() < 0 or tch.max() > 1) else tch
            print(f"    teacher p AUC {roc_auc_score(y_tr, p_in):.5f}", flush=True)
            soft_t = (1 - args.soft) * y_tr + args.soft * p_in
        elif args.soft > 0 and args.soft_inner_nn:
            # honest born-again teacher: inner 3-fold RealMLP on the fold's synthetic training rows
            p_in = np.zeros(n_syn)
            for ia, ib in StratifiedKFold(3, shuffle=True, random_state=fold).split(x_tr, y_tr):
                f_a, f_b = to_frames((x_tr.iloc[ia], x_tr.iloc[ib]), cat_cols)
                tm = RealMLP_TD_Classifier(**{**cfg, "random_state": args.seed + fold + 100})
                tm.fit(f_a, y_tr[ia].astype(np.int64), f_b, y_tr[ib].astype(np.int64),
                       cat_col_names=cat_cols, ckpt_path=CKPT_DIR / f"inner_{fold}.pth", X_test=f_b)
                p_in[ib] = tm.predict_proba(f_b)[:, 1]
                del tm
                torch.cuda.empty_cache()
            print(f"    inner RealMLP teacher AUC {roc_auc_score(y_tr, p_in):.5f}", flush=True)
            soft_t = (1 - args.soft) * y_tr + args.soft * p_in
        elif args.soft > 0:
            from lightgbm import LGBMClassifier
            p_in = np.zeros(n_syn)
            for a_i, b_i in StratifiedKFold(3, shuffle=True, random_state=fold).split(x_tr, y_tr):
                gbm = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=127,
                                     min_child_samples=200, colsample_bytree=0.6, subsample=0.8,
                                     subsample_freq=1, reg_lambda=5.0, n_jobs=12, verbose=-1)
                gbm.fit(x_tr.iloc[a_i], y_tr[a_i])
                p_in[b_i] = gbm.predict_proba(x_tr.iloc[b_i])[:, 1]
            print(f"    inner p AUC {roc_auc_score(y_tr, p_in):.5f}", flush=True)
            soft_t = (1 - args.soft) * y_tr + args.soft * p_in
        if x_orig is not None:
            assert m.y_orig is not None
            x_tr = pd.concat([x_tr, *[x_o[x_tr.columns]] * args.origrep], ignore_index=True)
            y_tr = np.concatenate([y_tr, *[m.y_orig] * args.origrep])
        if args.resid:
            from run_lgbm_arm import route_residual
            x_tr, x_va, x_te = route_residual(x_tr, y_tr, [x_va, x_te])
        tr, va, te = to_frames((x_tr, x_va, x_te), cat_cols)
        model = RealMLP_TD_Classifier(**{**cfg, "random_state": args.seed + fold})
        psw = None
        if soft_t is not None:
            # each synthetic row twice: (x, 1) weighted t and (x, 0) weighted 1 - t; others hard
            syn = tr.iloc[:n_syn]
            tr = pd.concat([syn, tr], ignore_index=True)
            psw = np.concatenate([soft_t, 1 - soft_t, np.ones(len(y_tr) - n_syn)])
            y_tr = np.concatenate([np.ones(n_syn), np.zeros(n_syn), y_tr[n_syn:]])
            psw = psw * len(psw) / psw.sum() / 1.0
        model.fit(
            tr,
            y_tr.astype(np.int64),
            va,
            y_va.astype(np.int64),
            cat_col_names=cat_cols,
            per_sample_weight=psw,
            # Per fold, or the five folds overwrite each other's best checkpoint.
            ckpt_path=CKPT_DIR / f"{args.tag or 'v1'}_s{args.seed}_k{args.nfolds}_f{fold}.pth",
            X_test=te,
        )
        val = np.asarray(model.best_val_probs_)[:, 1]
        test = model.predict_proba(te)[:, 1]
        if args.pseudo > 0:
            # Honest: the test labels come from this fold's model, which never saw the val fold.
            print(f"    stage1 val AUC {roc_auc_score(y_va, val):.5f}", flush=True)
            del model
            torch.cuda.empty_cache()
            rng = np.random.default_rng(args.seed + fold)
            pick = rng.random(len(te)) < args.pseudo
            y_pl = (rng.random(int(pick.sum())) < test[pick]).astype(np.int64)
            tr2 = pd.concat([tr, te[pick]], ignore_index=True)
            y2 = np.concatenate([y_tr.astype(np.int64), y_pl])
            model = RealMLP_TD_Classifier(**{**cfg, "random_state": args.seed + fold})
            model.fit(tr2, y2, va, y_va.astype(np.int64), cat_col_names=cat_cols,
                      ckpt_path=CKPT_DIR / f"{args.tag or 'v1'}_s{args.seed}_k{args.nfolds}_f{fold}_pl.pth",
                      X_test=te)
            val = np.asarray(model.best_val_probs_)[:, 1]
            test = model.predict_proba(te)[:, 1]
            del tr2
        print(
            f"  fold {fold}: AUC {roc_auc_score(y_va, val):.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del model, tr, va, te
        torch.cuda.empty_cache()
        return val, test

    if args.fullfit:
        # No validation is used for selection with epoch_select="last", so the "val" frame
        # is a slice of the training rows, there only to satisfy the fit API.
        if cfg.get("epoch_select") != "last":
            raise SystemExit("--fullfit needs epoch_select=last (nothing to early-stop on)")
        _, test = fit_fold(X, m.y, X.iloc[:2000], m.y[:2000], X_test, 0)
        run_name = f"realmlp_{args.tag or 'v1'}" + ("" if args.seed == 42 else f"_s{args.seed}")
        np.save(RESULTS_DIR / f"test_{run_name}_full.npy", test)
        print(f"full-fit test → test_{run_name}_full.npy", flush=True)
        return

    if args.fold1:
        # Same splitter and seed as run_cv, so this is fold 1 of a full run.
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, va_idx = next(iter(splitter.split(X, m.y)))
        val, _ = fit_fold(
            X.iloc[tr_idx], m.y[tr_idx], X.iloc[va_idx], m.y[va_idx], X_test.iloc[:1000], 1
        )
        np.save(RESULTS_DIR / f"probe_realmlp_{args.tag or 'v1'}_f1.npy",
                np.stack([va_idx.astype(float), val]))
        print(f"FOLD1 tag={args.tag or 'v1'} seed={args.seed} {overrides} "
              f"AUC {roc_auc_score(m.y[va_idx], val):.5f}", flush=True)
        return

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(
        X, m.y, X_test, fit_fold, n_splits=args.nfolds, n_outputs=1, score_fn=fold_score,
        after_fold=lambda _f: torch.cuda.empty_cache(),
    )
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"\nfolds {[round(s, 5) for s in fold_scores]}  OOF AUC: {oof_auc:.5f}")

    run_name = f"realmlp_{args.tag or 'v1'}" + ("" if args.seed == 42 else f"_s{args.seed}")
    run_name += "" if args.nfolds == 5 else f"_k{args.nfolds}"
    run_name += "" if args.pseudo == 0 else f"_pl{args.pseudo:g}"
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test)
    out = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", m.test_ids, TARGET, test)
    print(f"Arrays → {RESULTS_DIR}   Submission → {out}")


if __name__ == "__main__":
    main()
