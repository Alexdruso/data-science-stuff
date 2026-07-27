"""Day-20: how optimistic are the OOF arrays, per leg? (selection-on-the-scored-rows)

Two of the four u-family legs pick their *stopping point* on the very rows whose
predictions become the OOF:

  * realmlp  — realmlp.py's fit loop keeps `best_val_probs_` = argmax over epochs of
               balanced_accuracy_score(y_v, argmax(val_probs)) on the VAL FOLD
               (train_realmlp_td.py:192 stores exactly that as the fold's OOF).
  * lgbm_te3 — `early_stopping(50)` with `eval_set=[(X_va, y_va)]` (train_lgbm_te3.py:91).

  * hgbc_te3 — sklearn HGBC `early_stopping=True` carves `validation_fraction` out of
               FOLD-TRAIN → honest.
  * catboost_te3 — fixed iterations, no eval_set → honest.

This is an ESTIMATION bias, not a leak into test predictions (test comes from the
restored checkpoint), but every instrument in the campaign (split-half seat tests,
LOFO, finals E[max] MCs, the LR meta) consumes the OOF arrays as ground truth.

Protocol (one fold, seed 42, fold 1, u surface, deployed recipes, cached TE blocks):
  DEPLOYED  pick the stopping point on A u B, score BA on A u B   (what ships)
  HONEST    pick on A, score on B  (and B -> A)                   (unbiased)
  CONTROL   pick on A, score on A                                 (in-sample at the
                                                                   HONEST selection size)
`CONTROL - HONEST` isolates the bias at MATCHED selection-set size; because selection
optimism shrinks with the size of the selection set, it is an UPPER BOUND on the
deployed (full-val-selection) optimism. `DEPLOYED - HONEST` is the headline read.

Experiment C (realmlp only): re-adjudicate the epoch-selection RULES honestly
(select on A, score on B). Both Day-10 verdicts (`bll` NEGATIVE, `snapshot_k` WASH)
were scored on an OOF metric that IS the max of val-fold bacc, which structurally
favours the incumbent by about the size of the effect being tested.

READ-ONLY: writes only `results/probe_oof_selection*.{txt,npz}`; no oof_*/test_*/
decision_weights_*/submissions artifact is touched.

Run:
  python src/probe_oof_selection.py --arm lgbm      # CPU, n_jobs<=3
  python src/probe_oof_selection.py --arm realmlp   # GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, RESULTS_DIR, load_dataset
from zoo_common import te13_block_for_fold

warnings.filterwarnings("ignore", category=FutureWarning)

FOLD = 1
SEED = 42
SPLIT_SEED = 20260724
OUT = RESULTS_DIR / "probe_oof_selection.txt"  # per-arm suffix applied in main()


# --------------------------------------------------------------------------- utils
def fold_indices(y: NDArray[np.int64], seed: int, fold: int) -> tuple[NDArray, NDArray]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(y, y), 1):
        if i == fold:
            return tr, va
    raise ValueError(fold)


N_SPLITS = 20  # independent stratified A/B halvings of the val fold


def half_split(y_val: NDArray[np.int64], rep: int = 0) -> tuple[NDArray, NDArray]:
    """Stratified halves A / B of the val fold (positional indices into the val fold)."""
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SPLIT_SEED + rep)
    a, b = next(iter(skf.split(y_val, y_val)))
    return a, b


def bacc(y: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, probs.argmax(axis=1)))


def bll(y: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    """Balanced logloss (mean over classes of that class's mean NLL). Lower is better."""
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0))
    return float(np.mean([nll[y == c].mean() for c in np.unique(y)]))


def mlogloss(y: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    return float(np.mean(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))))


def es_best(curve: NDArray[np.float64], patience: int = 50) -> int:
    """LightGBM early_stopping(patience) emulation on a LOWER-is-better curve.

    Returns the 0-based index of the retained best iteration.
    """
    best_i, best_v = 0, curve[0]
    for i in range(1, len(curve)):
        if curve[i] < best_v:
            best_i, best_v = i, curve[i]
        elif i - best_i >= patience:
            break
    return best_i


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Tee:
    def __init__(self, path: Path) -> None:
        self.lines: list[str] = []
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def flush_to_disk(self) -> None:
        prev = self.path.read_text() if self.path.exists() else ""
        self.path.write_text(prev + "\n".join(self.lines) + "\n")


# ------------------------------------------------------------------------ lgbm arm
def run_lgbm(say: Tee) -> None:
    from lightgbm import LGBMClassifier, log_evaluation

    N_EST = 1000
    STEP = 5

    ds = load_dataset()
    y = ds.y
    tr_idx, va_idx = fold_indices(y, SEED, FOLD)
    y_va = y[va_idx]
    a_pos, b_pos = half_split(y_va)
    say(f"[lgbm] fold {FOLD} seed {SEED}: n_train={len(tr_idx)} n_val={len(va_idx)}")
    say(f"[lgbm] halves A={len(a_pos)} B={len(b_pos)} (stratified, seed {SPLIT_SEED})")

    # cached te13 block == train_lgbm_te3.py's inline TargetEncoder recipe
    # (same 13 string-cast cols, smooth='auto', multiclass, seeded inner CV)
    t0 = time.time()
    te_tr_full, _te_test, te13_names = te13_block_for_fold(ds, tr_idx, SEED, FOLD, "u")
    say(f"[lgbm] te13 block from cache: {te_tr_full.shape} ({time.time() - t0:.0f}s)")

    raw_cols = ds.feature_cols
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    assert len(te_names) == len(te13_names) == te_tr_full.shape[1]

    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")

    def assemble(pos: NDArray[np.int64]) -> pd.DataFrame:
        return pd.concat(
            [
                X.iloc[pos].reset_index(drop=True),
                pd.DataFrame(te_tr_full[pos], columns=te_names),
            ],
            axis=1,
        )

    X_tr, X_va = assemble(tr_idx), assemble(va_idx)

    from lgbm_device import get_lgbm_device
    from train_lgbm_te3 import LGBM_TE3_PARAMS

    device_type, _ = get_lgbm_device()
    params = {
        **LGBM_TE3_PARAMS,
        "n_estimators": N_EST,
        "n_jobs": 3,  # another agent holds the CPU lane
        "device_type": device_type,
        "random_state": SEED,
    }
    say(f"[lgbm] params: {json.dumps({k: str(v) for k, v in params.items()})}")

    t0 = time.time()
    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y[tr_idx],
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[log_evaluation(0)],
    )
    say(f"[lgbm] fit {N_EST} iters in {time.time() - t0:.0f}s")

    # per-iteration full-val logloss comes free from eval_set
    full_ll_curve = np.asarray(model.evals_result_["valid_0"]["multi_logloss"])
    k_es = es_best(full_ll_curve) + 1
    say(f"[lgbm] emulated early_stopping(50) on full val -> best_iteration={k_es}")

    # deployed reproduction check: predict at the ES iteration
    p_es = model.predict_proba(X_va, num_iteration=k_es)
    say(
        f"[lgbm] REPRODUCTION: fold-1 argmax BA at ES iter = {bacc(y_va, p_es):.6f} "
        "(recorded lgbm_te3_u_s42 fold_1 = 0.948853)"
    )

    # incremental raw scores over a step-STEP grid (raw scores are additive over trees)
    booster = model.booster_
    grid = list(range(STEP, N_EST + 1, STEP))
    raw = np.zeros((len(va_idx), N_CLASSES), dtype=np.float64)
    probs_grid = np.zeros((len(grid), len(va_idx), N_CLASSES), dtype=np.float32)
    t0 = time.time()
    for gi, k in enumerate(grid):
        raw += booster.predict(
            X_va, start_iteration=k - STEP, num_iteration=STEP, raw_score=True
        )
        probs_grid[gi] = softmax(raw).astype(np.float32)
    say(f"[lgbm] grid of {len(grid)} iteration counts scored in {time.time() - t0:.0f}s")

    # additivity check against a direct predict at the last grid point
    direct = model.predict_proba(X_va, num_iteration=grid[-1])
    say(f"[lgbm] additivity check max|Δp| = {np.abs(direct - probs_grid[-1]).max():.2e}")

    _selection_report(say, "lgbm_te3", y_va, probs_grid, grid, "iters")

    np.savez_compressed(
        RESULTS_DIR / "probe_oof_selection_lgbm.npz",
        grid=np.asarray(grid),
        y_val=y_va,
        grid_pred=probs_grid.argmax(axis=2).astype(np.int8),
        full_ll_curve=full_ll_curve,
    )


# --------------------------------------------------------------------- realmlp arm
def run_realmlp(say: Tee) -> None:
    import torch

    import data_science_stuff.kaggle.models.realmlp as rm
    from train_mlp import NNData
    from zoo_common import te_block_for_fold

    from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

    ds = load_dataset()
    y = ds.y
    tr_idx, va_idx = fold_indices(y, SEED, FOLD)
    y_va = y[va_idx]
    a_pos, b_pos = half_split(y_va)
    say(f"[realmlp] fold {FOLD} seed {SEED}: n_train={len(tr_idx)} n_val={len(va_idx)}")
    say(f"[realmlp] halves A={len(a_pos)} B={len(b_pos)}")

    nd = NNData(ds)
    cat_names = [f"cat_{c}" for c in nd.cat_cols]
    miss_names = [f"miss_{i}" for i in range(nd.miss_tr.shape[1])]

    te_tr_full, _te_te, _n = te_block_for_fold(ds, tr_idx, SEED, FOLD, "u")
    t13_tr, _t13_te, _n13 = te13_block_for_fold(ds, tr_idx, SEED, FOLD, "u")
    te_tr_full = np.concatenate([te_tr_full, t13_tr], axis=1)
    say(f"[realmlp] recipe block (te + te13) from cache: {te_tr_full.shape}")

    num_all, miss_all, cat_all = nd.Xnum_tr, nd.miss_tr, nd.Xcat_tr
    num_med = np.nanmedian(num_all[tr_idx], axis=0)
    num_f = np.where(np.isnan(num_all), num_med, num_all)
    te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
    te_f = np.where(np.isnan(te_tr_full), te_med, te_tr_full)

    X_all = pd.DataFrame(
        {
            **{n: cat_all[:, i] for i, n in enumerate(cat_names)},
            **{n: num_f[:, i] for i, n in enumerate(nd.num_cols)},
            **{n: miss_all[:, i] for i, n in enumerate(miss_names)},
            **{f"te_{i}": te_f[:, i] for i in range(te_f.shape[1])},
        }
    )
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[va_idx]

    # capture every epoch's val_probs from the fit loop (default recipe, unmodified):
    # the loop scores each epoch with balanced_accuracy_score(y_v, argmax(val_probs))
    # and `val_probs` is a local of that frame.
    captured: list[NDArray[np.float64]] = []
    orig_bas = rm.balanced_accuracy_score

    def spy(y_true, y_pred, **kw):  # noqa: ANN001, ANN202
        frame = sys._getframe(1)
        vp = frame.f_locals.get("val_probs")
        if vp is not None:
            captured.append(np.asarray(vp, dtype=np.float64).copy())
        return orig_bas(y_true, y_pred, **kw)

    fold_seed = SEED + FOLD * 100
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)  # noqa: NPY002
    cfg = dict(
        train_bs=512,
        fused_optimizer=True,
        allow_amp=True,
        verbosity=2,
        epoch_select="bacc",
        snapshot_k=0,
        random_state=fold_seed,
    )
    model = RealMLP_TD_Classifier(**cfg)
    rm.balanced_accuracy_score = spy
    t0 = time.time()
    try:
        model.fit(X_tr, y[tr_idx], X_val, y_va, cat_col_names=cat_names)
    finally:
        rm.balanced_accuracy_score = orig_bas
    say(f"[realmlp] fit in {time.time() - t0:.0f}s; {len(captured)} epochs captured")

    P = np.stack(captured)  # (E, n_val, 3)
    ep_scores = np.asarray([bacc(y_va, P[e]) for e in range(len(P))])
    say(f"[realmlp] per-epoch full-val BA: {np.round(ep_scores, 6).tolist()}")
    say(
        f"[realmlp] model.best_score_={model.best_score_:.6f} "
        f"max(per-epoch)={ep_scores.max():.6f}  (must match)"
    )
    dep = bacc(y_va, model.best_val_probs_)
    say(
        f"[realmlp] REPRODUCTION: deployed fold-1 OOF BA = {dep:.6f} "
        "(recorded realmlp_te13_u_s42 fold_1 = 0.951068)"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    grid = list(range(1, len(P) + 1))
    _selection_report(say, "realmlp_te13", y_va, P, grid, "epochs")
    _experiment_c(say, y_va, P)

    np.savez_compressed(
        RESULTS_DIR / "probe_oof_selection_realmlp.npz",
        epoch_probs=P.astype(np.float32),
        y_val=y_va,
    )


# ------------------------------------------------------------------- shared reads
def _curves(
    corr: NDArray[np.bool_],
    nll: NDArray[np.float32],
    y_va: NDArray[np.int64],
    pos: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(BA curve, mean-NLL curve) over the candidate grid, restricted to rows `pos`."""
    yp = y_va[pos]
    recalls = [corr[:, pos[yp == c]].mean(axis=1) for c in np.unique(y_va)]
    return np.mean(recalls, axis=0), nll[:, pos].mean(axis=1)


def _selection_report(
    say: Tee,
    leg: str,
    y_va: NDArray[np.int64],
    probs_grid: NDArray[np.float32],
    grid: list[int],
    unit: str,
) -> None:
    """DEPLOYED / HONEST / CONTROL for both selection criteria, over N_SPLITS halvings."""
    pred = probs_grid.argmax(axis=2)
    corr = pred == y_va[None, :]
    nll = -np.log(
        np.clip(probs_grid[:, np.arange(len(y_va)), y_va], 1e-15, 1.0)
    ).astype(np.float32)
    all_pos = np.arange(len(y_va))
    ba_full, ll_full = _curves(corr, nll, y_va, all_pos)

    for crit in ("bacc", "mlogloss"):
        if crit == "bacc":
            i_full = int(ba_full.argmax())
        else:
            i_full = es_best(ll_full)
        dep = float(ba_full[i_full])
        rows = []
        for rep in range(N_SPLITS):
            a_pos, b_pos = half_split(y_va, rep)
            ba_a, ll_a = _curves(corr, nll, y_va, a_pos)
            ba_b, ll_b = _curves(corr, nll, y_va, b_pos)
            if crit == "bacc":
                i_a, i_b = int(ba_a.argmax()), int(ba_b.argmax())
            else:
                i_a, i_b = es_best(ll_a), es_best(ll_b)
            rows.append(
                (
                    grid[i_a],
                    grid[i_b],
                    float(ba_b[i_a]),
                    float(ba_a[i_b]),
                    float(ba_a[i_a]),
                    float(ba_b[i_b]),
                )
            )
        arr = np.asarray(rows, dtype=np.float64)
        hon = 0.5 * (arr[:, 2] + arr[:, 3])
        ctl = 0.5 * (arr[:, 4] + arr[:, 5])
        say("")
        say(f"--- {leg} | selection criterion: {crit}  ({N_SPLITS} A/B splits) ---")
        say(f"  DEPLOYED pick on AuB: {unit}={grid[i_full]}   BA(AuB) = {dep:.6f}")
        say(
            f"  honest-picked {unit}: A median={int(np.median(arr[:, 0]))} "
            f"B median={int(np.median(arr[:, 1]))}"
        )
        say(f"  HONEST  mean over splits : {hon.mean():.6f}  (sd {hon.std(ddof=1):.6f})")
        say(f"  CONTROL mean over splits : {ctl.mean():.6f}  (sd {ctl.std(ddof=1):.6f})")
        d_dh = dep - hon
        d_ch = ctl - hon
        say(
            f"  >> DEPLOYED - HONEST     : {d_dh.mean():+.6f}  "
            f"(sd {d_dh.std(ddof=1):.6f}, se {d_dh.std(ddof=1) / np.sqrt(N_SPLITS):.6f}, "
            f"{int((d_dh > 0).sum())}/{N_SPLITS} positive)"
        )
        say(
            f"  >> CONTROL - HONEST      : {d_ch.mean():+.6f}  "
            f"(sd {d_ch.std(ddof=1):.6f}, se {d_ch.std(ddof=1) / np.sqrt(N_SPLITS):.6f}, "
            f"{int((d_ch > 0).sum())}/{N_SPLITS} positive)   "
            "[matched selection-set size -> UPPER BOUND on the deployed bias]"
        )


def _experiment_c(
    say: Tee, y_va: NDArray[np.int64], P: NDArray[np.float64]
) -> None:
    """Honest (select-on-A, score-on-B) re-adjudication of the epoch-selection rules."""
    n_ep = len(P)

    def rules(sel_pos, sco_pos):  # noqa: ANN001, ANN202
        sel_y, sco_y = y_va[sel_pos], y_va[sco_pos]
        ba_sel = np.asarray([bacc(sel_y, P[e][sel_pos]) for e in range(n_ep)])
        bll_sel = np.asarray([bll(sel_y, P[e][sel_pos]) for e in range(n_ep)])
        out: dict[str, float] = {"bacc": bacc(sco_y, P[int(ba_sel.argmax())][sco_pos])}
        out["bll"] = bacc(sco_y, P[int(bll_sel.argmin())][sco_pos])
        for k in (2, 3, 4):
            top = np.argsort(-ba_sel, kind="stable")[:k]
            out[f"snap{k}"] = bacc(sco_y, P[top].mean(axis=0)[sco_pos])
        out["last"] = bacc(sco_y, P[-1][sco_pos])
        return out

    per_rule: dict[str, list[float]] = {}
    for rep in range(N_SPLITS):
        a_pos, b_pos = half_split(y_va, rep)
        ab, ba_ = rules(a_pos, b_pos), rules(b_pos, a_pos)
        for rule in ab:
            per_rule.setdefault(rule, []).extend([ab[rule], ba_[rule]])
    say("")
    say(
        f"--- Experiment C: HONEST re-adjudication of epoch-selection rules "
        f"({N_SPLITS} splits x 2 directions) ---"
    )
    say("  rule      honest BA (mean)   delta vs bacc   se(delta)   n>0 / 40")
    base = np.asarray(per_rule["bacc"])
    for rule, vals in per_rule.items():
        v = np.asarray(vals)
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if rule != "bacc" else 0.0
        say(
            f"  {rule:<8}  {v.mean():.6f}          {d.mean():+.6f}      "
            f"{se:.6f}    {int((d > 0).sum())}"
        )


# --------------------------------------------------------------------- ledger arm
def run_ledger(say: Tee, only: str = "both") -> None:
    """Does the early-stopping rule actually FIRE for the other two u4 GBDT legs?

    `lgbm_bagw` (baseline.py: lr 0.05, 127 leaves) and `xgboost_bagw`
    (train_xgboost.py: lr 0.05, depth 8, 2000 rounds) both early-stop on the VAL
    FOLD, unlike the heavily-regularized lgbm_te3 config. If their mlogloss curve
    bottoms out before the cap, the stopping point IS selected on the scored rows.
    """
    import xgboost as xgb
    from lightgbm import LGBMClassifier, log_evaluation

    ds = load_dataset()
    y = ds.y
    tr_idx, va_idx = fold_indices(y, SEED, FOLD)
    y_va = y[va_idx]
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

    # ---- lgbm_bagw (baseline.py recipe)
    from baseline import LGBM_PARAMS

    p = {**LGBM_PARAMS, "n_jobs": 3, "random_state": SEED}
    t0 = time.time()
    m = LGBMClassifier(**p)
    m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y_va)], callbacks=[log_evaluation(0)])
    curve = np.asarray(m.evals_result_["valid_0"]["multi_logloss"])
    k_es = es_best(curve) + 1
    say("")
    say(
        f"[ledger] lgbm_bagw (baseline cfg, {p['n_estimators']} cap): "
        f"ES(50) fires at iter {k_es} "
        f"({'FIRES -> selection on the scored rows' if k_es < p['n_estimators'] else 'never fires -> honest'})"
        f"  [{time.time() - t0:.0f}s]"
    )
    step = max(1, k_es // 100)
    grid = list(range(step, min(int(p["n_estimators"]), k_es + 200) + 1, step))
    raw = np.zeros((len(va_idx), N_CLASSES))
    pg = np.zeros((len(grid), len(va_idx), N_CLASSES), dtype=np.float32)
    for gi, k in enumerate(grid):
        raw += m.booster_.predict(
            X_va, start_iteration=k - step, num_iteration=step, raw_score=True
        )
        pg[gi] = softmax(raw).astype(np.float32)
    _selection_report(say, "lgbm_bagw", y_va, pg, grid, "iters")
    del m, pg, raw

    # ---- xgboost_bagw (train_xgboost.py recipe)
    from train_xgboost import load_params

    xp = dict(load_params())
    n_cap = int(xp.pop("n_estimators"))
    xp.pop("early_stopping_rounds", None)
    xp["random_state"] = SEED
    t0 = time.time()
    xm = xgb.XGBClassifier(n_estimators=n_cap, **xp)
    xm.fit(X_tr, y[tr_idx], eval_set=[(X_va, y_va)], verbose=False)
    xcurve = np.asarray(xm.evals_result()["validation_0"]["mlogloss"])
    k_es = es_best(xcurve) + 1
    say("")
    say(
        f"[ledger] xgboost_bagw (baseline cfg, {n_cap} cap): ES(50) fires at iter {k_es} "
        f"({'FIRES -> selection on the scored rows' if k_es < n_cap else 'never fires -> honest'})"
        f"  [{time.time() - t0:.0f}s]"
    )
    # XGB margins carry base_score per call, so incremental sums are unsafe here:
    # predict the cumulative range directly at a coarser grid instead.
    hi = min(n_cap, k_es + 300)
    step = max(5, hi // 60)
    grid = list(range(step, hi + 1, step))
    pg = np.zeros((len(grid), len(va_idx), N_CLASSES), dtype=np.float32)
    t0 = time.time()
    for gi, k in enumerate(grid):
        pg[gi] = xm.predict_proba(X_va, iteration_range=(0, k)).astype(np.float32)
    say(f"[ledger] xgb grid of {len(grid)} points in {time.time() - t0:.0f}s")
    _selection_report(say, "xgboost_bagw", y_va, pg, grid, "iters")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lgbm", "realmlp", "ledger"], required=True)
    args = ap.parse_args()
    say = Tee(OUT.with_name(f"probe_oof_selection_{args.arm}.txt"))
    say("=" * 78)
    say(f"probe_oof_selection — arm={args.arm}  ({time.strftime('%Y-%m-%d %H:%M')})")
    say("=" * 78)
    try:
        {"lgbm": run_lgbm, "realmlp": run_realmlp, "ledger": run_ledger}[args.arm](say)
    finally:
        say.flush_to_disk()


if __name__ == "__main__":
    main()
