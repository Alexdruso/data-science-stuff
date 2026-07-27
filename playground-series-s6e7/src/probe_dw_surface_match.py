"""Pre-registration #4, cell B: the decision layer is TUNED on a 1-model-per-row surface
and APPLIED to a 5-model-average surface. Does that mismatch cost anything?

Deployed asymmetry, unchanged all campaign:
  OOF row i  -> predicted by the ONE fold model whose val fold contains i
  test row   -> predicted by the MEAN of all 5 fold models
and `robust_decision_weights` is tuned on the former, then applied to the latter.

Day-20 raised this and closed it on a PROXY — "max-p gap -0.00117, wrong sign => no
mechanism, the nested-CV follow-up is unnecessary". That proxy measures *calibration
sharpness*. The quantity that actually matters is *per-row prediction noise*: averaging
M models shrinks it whatever it does to mean confidence, and an argmax metric responds to
it by flipping boundary rows. Day-21's 1/M result is a direct demonstration that per-row
noise drives argmax-metric deltas in ways a confidence proxy does not see — so the
closure is reopened and measured.

DESIGN (nested, honest, no finals lineage touched)
  For each outer fold k (deployed 5-fold split, seed 42):
    * build the fold's te13 block ONCE (fit on fold-train) and hold it fixed, so the
      variable under test is model-averaging alone, not TE-averaging;
    * split fold-train into 5 inner folds; train 5 sub-models, each on 4/5 of fold-train;
    * SURFACE_M (matched)  = mean of the 5 sub-models' predictions on the val fold
      -> val rows scored by a 5-model average of models that never saw them, i.e. the
         same averaging structure the TEST surface has;
    * SURFACE_D (deployed) = one model trained on the whole fold-train, predicting the
      val fold -> exactly the deployed OOF construction.

  Then, on the held-out half of a split-half protocol (10 stratified halvings):
    w_D = weights tuned on SURFACE_D (fit half)      <- what ships
    w_M = weights tuned on SURFACE_M (fit half)      <- matched to the applied surface
    score BOTH on SURFACE_M (score half)             <- the test-like surface
  HEADLINE = BA(SURFACE_M, w_M) - BA(SURFACE_M, w_D)  = what the mismatch costs.

  Sub-models see 64% of the data vs the deployed 80%, so SURFACE_M's LEVEL is not
  comparable to SURFACE_D's. That is why the headline compares two weight vectors on the
  SAME surface — the level confound cancels exactly.

PRE-REGISTERED BARS (CLAUDE.md "PRE-REGISTRATION #4"), raised by the freeze because any
adoption would change the deployed, byte-verified, already-submitted #1:
  >= +0.0005  report to the user as a PROPOSAL to reopen the freeze, with costs stated
  <= +0.0001  cell CLOSED BY MEASUREMENT (upgrades Day-20's proxy-closure)
  otherwise   inconclusive, no action

Honest prior: ~0. Day-20 measured the BA surface in w-space as nearly flat (w vectors
differing by 0.2 score within 2e-5).

Run: python src/probe_dw_surface_match.py -> results/probe_dw_surface_match.txt
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, RESULTS_DIR, load_dataset, robust_decision_weights
from train_hgbc_te3 import HGBC_CONFIG

from data_science_stuff.kaggle_utils import weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
N_OUTER = 5
N_INNER = 5
N_HALVES = 10
SPLIT_SEED = 20260727
BAR_PROPOSE = 0.0005
BAR_CLOSE = 0.0001


def bacc(y: NDArray[np.int64], probs: NDArray[np.float64], w: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, weighted_predict(probs, w)))


def main() -> None:
    ds = load_dataset()
    y = ds.y
    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X[col] = pd.Categorical(X[col], categories=sorted(X[col].unique()))
    te_view = X[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    surf_d = np.zeros((len(y), N_CLASSES))
    surf_m = np.zeros((len(y), N_CLASSES))

    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
        t0 = time.time()
        enc = TargetEncoder(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            smooth="auto",
            target_type="multiclass",
        )
        te_tr = enc.fit_transform(te_view.iloc[tr_idx], y[tr_idx])
        te_va = enc.transform(te_view.iloc[va_idx])

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
            )

        X_tr_full = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[va_idx], te_va)

        # SURFACE_D — the deployed construction: one model on the whole fold-train
        m = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
        m.fit(X_tr_full, y[tr_idx])
        surf_d[va_idx] = m.predict_proba(X_va)
        del m

        # SURFACE_M — matched: mean of 5 sub-models, each on 4/5 of fold-train, none of
        # which saw any val row. Same averaging structure as the deployed TEST surface.
        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + fold)
        acc = np.zeros((len(va_idx), N_CLASSES))
        for sub_tr, _sub_va in inner.split(np.zeros(len(tr_idx)), y[tr_idx]):
            sm = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
            sm.fit(X_tr_full.iloc[sub_tr], y[tr_idx][sub_tr])
            acc += sm.predict_proba(X_va)
            del sm
        surf_m[va_idx] = acc / N_INNER
        print(f"fold {fold}: surfaces built in {time.time() - t0:.0f}s")

    lines = [
        "=" * 78,
        "probe_dw_surface_match — decision weights are tuned on a 1-model-per-row",
        "surface and applied to a 5-model-average one. What does the mismatch cost?",
        f"hgbc_te3 canary, seed {SEED}, {N_OUTER} outer x {N_INNER} inner, "
        f"{N_HALVES} stratified halvings",
        "te13 block held fixed per fold => the variable under test is model-averaging alone",
        f"PRE-REG BARS: >=+{BAR_PROPOSE} propose reopening the freeze / "
        f"<=+{BAR_CLOSE} closed by measurement",
        "=" * 78,
        "",
    ]

    w_full_d = robust_decision_weights(y, surf_d)
    w_full_m = robust_decision_weights(y, surf_m)
    lines += [
        "full-data reference (levels are NOT comparable — sub-models see 64% vs 80% of data):",
        f"  SURFACE_D  BA(w_D) = {bacc(y, surf_d, w_full_d):.6f}   w_D = {np.round(w_full_d, 4).tolist()}",
        f"  SURFACE_M  BA(w_M) = {bacc(y, surf_m, w_full_m):.6f}   w_M = {np.round(w_full_m, 4).tolist()}",
        f"  max|w_M - w_D| = {np.abs(np.asarray(w_full_m) - np.asarray(w_full_d)).max():.4f}",
        "",
        "HONEST split-half: tune on the fit half of each surface, score BOTH on SURFACE_M",
        "  (same surface, same rows => the 64%-vs-80% level confound cancels exactly)",
        "",
    ]
    print("\n".join(lines))

    deltas = []
    for rep in range(N_HALVES):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SPLIT_SEED + rep)
        for a, b in skf.split(np.zeros(len(y)), y):
            wd = robust_decision_weights(y[a], surf_d[a])
            wm = robust_decision_weights(y[a], surf_m[a])
            s_d = bacc(y[b], surf_m[b], wd)
            s_m = bacc(y[b], surf_m[b], wm)
            deltas.append(s_m - s_d)
            row = (
                f"  rep {rep} half {len(deltas) % 2}: w_D->{s_d:.6f}  w_M->{s_m:.6f}  "
                f"delta {s_m - s_d:+.6f}"
            )
            lines.append(row)
            print(row)

    d = np.asarray(deltas)
    head = (
        f"HEADLINE  BA(surface_M, w_M) - BA(surface_M, w_D) = {d.mean():+.6f}  "
        f"(sd {d.std(ddof=1):.6f}, se {d.std(ddof=1) / np.sqrt(len(d)):.6f}, "
        f"{int((d > 0).sum())}/{len(d)} positive)"
    )
    lines += ["", head]
    if d.mean() >= BAR_PROPOSE:
        lines.append("VERDICT: PROPOSE reopening the freeze — report to the user, do not act.")
    elif d.mean() <= BAR_CLOSE:
        lines.append(
            "VERDICT: CLOSED BY MEASUREMENT — the tune/apply surface mismatch costs nothing. "
            "Day-20's proxy-closure is upgraded to a measured one. Finals unchanged."
        )
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no action.")
    print(head)
    print(lines[-1])
    (RESULTS_DIR / "probe_dw_surface_match.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
