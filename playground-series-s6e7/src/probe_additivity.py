"""Day-20 pre-registered ADDITIVITY probe — do the measured sub-bar micro-positives ADD?

Every screen in this campaign is PER-LEVER with a +0.0007 (canary) / +0.001
(adoption) bar, inherited from the LB-blind protocol's RANKABILITY argument (no
instrument can rank sub-0.001 gaps, so never spend a submission on one). That
argument governs CHOOSING BETWEEN candidates; it says nothing about ACCUMULATING
unbiased same-artifact improvements. Nothing in the log has tested whether the
measured micro-positives on the hgbc_te3 consumer add.

Ledger (verified against results/ before running):
  basin config  (results/log_hgbc_te3_basin.txt, `_u_s42t`)   +0.0001  wt
  TE smooth=5   (results/probe_te_design.txt)                 +0.00008 wt
  IMP block     (results/probe_impute_driver{,_s7}.txt)       +0.00019 / +0.00011 wt
                (+0.00013 wt on the realmlp consumer, Day-17)
Nominal sum ~ +0.00037.  K=10 folds EXCLUDED (read -0.00004 on this consumer).

Design: ONE arm carrying ALL THREE simultaneously, paired in-script against a
stock hgbc_te3 arm on identical fold splits, 2 seeds (42, 7).  TE is computed
inline per (arm, fold) — no shared TE cache, so the smooth=5 encoding can never
collide with a cached smooth='auto' block (the campaign has been bitten by
cache-key collisions across surfaces).  Fold-model val probabilities are the
only cached objects, under fresh `ckpt_additivity_*` dirs.

PRE-REGISTERED BARS (2-seed mean weighted-OOF delta):
  >= +0.00025  ⇒ ADDITIVE (sum-of-parts minus ~1 sigma) — escalate (price the
                 realmlp arm; do not run it here).
  <= +0.0001   ⇒ additivity REFUTED; the portfolio lens dies.
  in between   ⇒ inconclusive, report as such.

Run: python src/probe_additivity.py            (seed 42)
     S6E7_PROBE_SEED=7 python src/probe_additivity.py
     python src/probe_additivity.py --report    (both seeds -> report)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    N_CLASSES,
    N_FOLDS,
    RESULTS_DIR,
    load_dataset,
    robust_decision_weights,
)
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle_utils import weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = int(os.environ.get("S6E7_PROBE_SEED", "42"))
SEEDS_ALL = (42, 7)
DRIVER = "sleep_duration"

# Day-12 TPE second basin, verbatim from results/log_hgbc_te3_basin.txt (the
# S6E7_HGBC_PARAMS override that produced hgbc_te3_u_s42t).
BASIN: dict[str, object] = {
    "learning_rate": 0.03373693124435465,
    "max_iter": 600,
    "max_leaf_nodes": 77,
    "min_samples_leaf": 391,
    "l2_regularization": 0.09110510408304606,
    "max_features": 0.40895869469862856,
    "max_bins": 192,
    "max_depth": None,
}

ARMS: dict[str, dict[str, object]] = {
    "base": {"config": dict(HGBC_CONFIG), "smooth": "auto", "imp": False},
    "all3": {"config": {**HGBC_CONFIG, **BASIN}, "smooth": 5.0, "imp": True},
}


def ckpt_dir(seed: int) -> Path:
    return CACHE_DIR / f"ckpt_additivity_s{seed}"


def run_seed(seed: int) -> dict[str, np.ndarray]:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        cats = sorted(
            set(X[col].unique())
            | set(ds.test[col].fillna("missing").astype(str).unique())
        )
        X[col] = pd.Categorical(X[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    other_cols = [c for c in raw_cols if c != DRIVER]
    sleep = ds.train[DRIVER].to_numpy(dtype=np.float64)
    obs = ~np.isnan(sleep)

    def assemble(
        base: pd.DataFrame,
        blk: np.ndarray,
        extra: "dict[str, np.ndarray] | None" = None,
    ) -> pd.DataFrame:
        parts = [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)]
        if extra:
            parts.append(pd.DataFrame(extra))
        return pd.concat(parts, axis=1)

    ckpt = ckpt_dir(seed)
    ckpt.mkdir(parents=True, exist_ok=True)
    oofs = {a: np.zeros((len(ds.y), N_CLASSES)) for a in ARMS}
    r2s: list[float] = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        done = {a: ckpt / f"{a}_f{fold}_val.npy" for a in ARMS}
        if all(p.exists() for p in done.values()):
            for a, p in done.items():
                oofs[a][val_idx] = np.load(p)
            print(f"  fold {fold}: loaded checkpoints", flush=True)
            continue

        extras: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        if not done["all3"].exists():
            # fold-safe driver reconstruction (probe_impute_driver recipe verbatim)
            fit_mask = np.zeros(len(ds.y), dtype=bool)
            fit_mask[tr_idx] = True
            fit_mask &= obs
            reg = HistGradientBoostingRegressor(
                max_iter=300,
                early_stopping=True,
                categorical_features="from_dtype",
                random_state=seed,
            )
            reg.fit(X.loc[fit_mask, other_cols], sleep[fit_mask])
            imp_all = reg.predict(X[other_cols]).astype(np.float64)
            val_obs = np.zeros(len(ds.y), dtype=bool)
            val_obs[val_idx] = True
            val_obs &= obs
            r2s.append(float(r2_score(sleep[val_obs], imp_all[val_obs])))
            print(f"  fold {fold}: imputer val R2 {r2s[-1]:.4f}", flush=True)
            completed = np.where(np.isnan(sleep), imp_all, sleep)
            extras["all3"] = (
                {"imp_sleep": imp_all[tr_idx], "sleep_completed": completed[tr_idx]},
                {"imp_sleep": imp_all[val_idx], "sleep_completed": completed[val_idx]},
            )

        for arm, spec in ARMS.items():
            if done[arm].exists():
                oofs[arm][val_idx] = np.load(done[arm])
                continue
            # seeded inner CV (Day-10/19 fix: sklearn's cv=<int> shuffles unseeded);
            # identical across arms, so the pairing is clean.
            enc = TargetEncoder(
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                smooth=spec["smooth"],
                target_type="multiclass",
            )
            te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
            te_va = enc.transform(te_tr_view.iloc[val_idx])
            ex_tr, ex_va = extras.get(arm, (None, None))
            model = HistGradientBoostingClassifier(
                **spec["config"], random_state=seed  # type: ignore[arg-type]
            )
            model.fit(assemble(X.iloc[tr_idx], te_tr, ex_tr), ds.y[tr_idx])
            val_proba = model.predict_proba(assemble(X.iloc[val_idx], te_va, ex_va))
            np.save(done[arm], val_proba)
            oofs[arm][val_idx] = val_proba
            score = balanced_accuracy_score(ds.y[val_idx], val_proba.argmax(axis=1))
            print(f"  [{arm}] fold {fold} argmax {score:.4f}", flush=True)
    if r2s:
        print(f"imputer val R2: {' '.join(f'{r:.4f}' for r in r2s)}", flush=True)
    return oofs


def boot_sd(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired row-bootstrap of the balanced-accuracy delta (b - a).

    Predictions are frozen (decision weights fitted once on full OOF), so this
    measures sampling noise of the DELTA on the same rows — the paired
    instrument's own SE, not the noise of an independent rerun.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    ok_a = (pred_a == y).astype(np.float64)
    ok_b = (pred_b == y).astype(np.float64)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        cnt = np.bincount(yb, minlength=N_CLASSES).astype(np.float64)
        ra = np.bincount(yb, weights=ok_a[idx], minlength=N_CLASSES) / cnt
        rb = np.bincount(yb, weights=ok_b[idx], minlength=N_CLASSES) / cnt
        deltas[i] = rb.mean() - ra.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.std(ddof=1)), float(lo), float(hi)


def report() -> None:
    ds = load_dataset()
    lines = [
        "ADDITIVITY PROBE (Day-20) — do the sub-bar micro-positives ADD?",
        "consumer hgbc_te3, u surface, K=5, paired in-script arms on identical folds",
        "  base = Susol v0.7 config + TE smooth='auto'",
        "  all3 = TPE basin config + TE smooth=5.0 + IMP driver-reconstruction block",
        "ledger (verified): basin +0.00010 / smooth5 +0.00008 / imp +0.00019,+0.00011",
        "nominal sum ~ +0.00037   |   PRE-REG BARS: >= +0.00025 additive, <= +0.0001 refuted",
        "",
    ]
    per_seed: list[tuple[int, float, float, float, float, float]] = []
    for seed in SEEDS_ALL:
        ck = ckpt_dir(seed)
        if not all(
            (ck / f"{a}_f{f}_val.npy").exists() for a in ARMS for f in range(1, 6)
        ):
            lines.append(f"seed {seed}: INCOMPLETE (checkpoints missing) — skipped")
            continue
        oofs = {a: np.zeros((len(ds.y), N_CLASSES)) for a in ARMS}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
            for a in ARMS:
                oofs[a][val_idx] = np.load(ck / f"{a}_f{fold}_val.npy")
        res = {}
        preds = {}
        for a, oof in oofs.items():
            am = float(balanced_accuracy_score(ds.y, oof.argmax(axis=1)))
            w = robust_decision_weights(ds.y, oof)
            pred = weighted_predict(oof, w)
            wt = float(balanced_accuracy_score(ds.y, pred))
            res[a] = (am, wt, w)
            preds[a] = pred
            lines.append(
                f"seed {seed}  {a:5s} argmax {am:.5f}  weighted {wt:.5f}  w={np.round(w, 4)}"
            )
        d_am = res["all3"][0] - res["base"][0]
        d_wt = res["all3"][1] - res["base"][1]
        sd, lo, hi = boot_sd(ds.y, preds["base"], preds["all3"], seed=seed)
        flips = int((preds["base"] != preds["all3"]).sum())
        lines.append(
            f"seed {seed}  PAIRED delta: argmax {d_am:+.5f}  weighted {d_wt:+.5f}"
            f"   boot sd {sd:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  flips {flips}"
        )
        lines.append("")
        per_seed.append((seed, d_am, d_wt, sd, lo, hi))

    if len(per_seed) == 2:
        mean_wt = float(np.mean([p[2] for p in per_seed]))
        mean_am = float(np.mean([p[1] for p in per_seed]))
        sd_mean = float(np.sqrt(sum(p[3] ** 2 for p in per_seed)) / 2)
        spread = abs(per_seed[0][2] - per_seed[1][2])
        lines += [
            f"2-SEED MEAN delta: weighted {mean_wt:+.5f} (argmax {mean_am:+.5f})",
            f"  paired boot sd of the mean {sd_mean:.5f}; between-seed spread {spread:.5f}",
        ]
        if mean_wt >= 0.00025:
            verdict = "ADDITIVE — micro-positives accumulate; escalate (price realmlp arm)"
        elif mean_wt <= 0.0001:
            verdict = "REFUTED — additivity does not hold; portfolio lens dies"
        else:
            verdict = "INCONCLUSIVE — between the pre-registered bars"
        lines.append(f"VERDICT: {verdict}")
    out = RESULTS_DIR / "probe_additivity.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nreport -> {out}", flush=True)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        print(f"[additivity] seed {SEED}", flush=True)
        run_seed(SEED)
