"""One-fold diagnostic: does the adopted XGBoost config have capacity left?

    python src/diag_capacity.py
    python src/diag_capacity.py --cap 10000 --fold 1

**The finding this exists to chase.** Optuna study `xgb_s1_v3` trial 8 — the point
adopted on 08-13 at +0.001139 — recorded `best_iteration = 2999` of
`n_estimators = 3000`. Early stopping at patience 100 **never fired**. The adopted
config was cut off, not converged. Two of its other parameters were likewise
selected at the edge of the box they were searched in (`max_bin = 512` is the top
of `{256, 512}`; the `learning_rate` floor of 0.015 was never breached).

**Why one fit answers two configs.** Gradient boosting is deterministic and
additive: with the same params, the same seed and the same rows, the first 3000
trees of a 10000-tree fit ARE the 3000-tree fit. So the shipped config is a strict
*prefix* of the raised-cap config, and reading the validation curve at round 3000
gives the shipped arm exactly — not approximately, and with no second fit and no
paired-noise to argue about. This is why the diagnostic reports a curve rather
than two numbers.

**What this is and is not.** It is mechanistic: "is there headroom at all?", a
question about where ES fires, which one fold answers definitively. It is NOT a
score read — fold sd here is ~0.0005 and the bar is +0.0005, so one fold cannot
resolve the bar and is not asked to. The pre-registered gate is
`best_iteration >= 3500`; the score read is the 3-seed paired 5-fold in
CLAUDE.md's Day-15 pre-registration.

**Read the tail shape, not just the endpoint.** A curve that gains +0.0008 between
rounds 3000 and 6000 is capacity. A curve that wanders +0.0001 across 4000 rounds
and happens to peak late is early stopping picking noise on the very fold it is
scored on — which is the confound the pre-registration names, and the reason the
adoption guard is a no-early-stopping refit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from run_arm import ARMS, build_matrices
from train_xgboost import PARAMS

from data_science_stuff.kaggle.io import competition_dirs, load_params

_, RESULTS_DIR, _ = competition_dirs(__file__)

SEED = 42
N_FOLDS = 5
#: The cap the adopted config ran into, and the arm it defines.
SHIPPED_CAP = 3000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1")
    parser.add_argument("--fold", type=int, default=1, help="which of the 5 folds (1-based)")
    parser.add_argument("--cap", type=int, default=10000, help="the raised n_estimators cap")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="early_stopping_rounds; matches the adopted config by default",
    )
    parser.add_argument("--params", default="best_params_xgboost.json")
    args = parser.parse_args()

    params = dict(load_params(RESULTS_DIR, PARAMS, args.params))
    params["n_jobs"] = PARAMS["n_jobs"]
    params["random_state"] = SEED
    params["n_estimators"] = args.cap
    params["early_stopping_rounds"] = args.patience
    params["eval_metric"] = "auc"

    x, y, _ = build_matrices(ARMS[args.arm][0])
    splits = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x, y)
    )
    tr_idx, va_idx = splits[args.fold - 1]
    x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    print(
        f"capacity diagnostic on arm {args.arm}, fold {args.fold}/{N_FOLDS}: "
        f"train {x_tr.shape}, val {x_va.shape}",
        flush=True,
    )
    print(
        f"  lr={params['learning_rate']:.6f} depth={params['max_depth']} "
        f"max_bin={params.get('max_bin')} cap={args.cap} patience={args.patience}\n",
        flush=True,
    )

    t0 = time.time()
    model = XGBClassifier(**params)
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    elapsed = time.time() - t0

    curve = np.asarray(model.evals_result_["validation_0"]["auc"], dtype=float)
    grown = len(curve)
    best_iter = int(model.best_iteration)

    # The shipped arm is the prefix. Its early stopping could only ever have
    # selected a round below its own cap, so its result is the best value the
    # curve attains there.
    prefix = curve[:SHIPPED_CAP]
    shipped_best = float(prefix.max())
    shipped_at = int(prefix.argmax())
    raised_best = float(curve.max())
    raised_at = int(curve.argmax())

    print(f"  fit {elapsed:.0f}s, grew {grown} trees, ES best_iteration={best_iter}")
    print("\n  validation AUC along the curve:")
    marks = [r for r in (500, 1000, 1500, 2000, 2500, 2999, 4000, 5000, 6000, 8000, 9999)
             if r < grown]
    for r in marks:
        flag = "  <- shipped cap" if r == SHIPPED_CAP - 1 else ""
        print(f"    round {r:>5}  {curve[r]:.5f}{flag}")

    print(f"\n  A  shipped  (cap {SHIPPED_CAP:>5})  best {shipped_best:.5f} at round {shipped_at}")
    print(f"  B  raised   (cap {args.cap:>5})  best {raised_best:.5f} at round {raised_at}")
    print(f"  fold delta B - A: {raised_best - shipped_best:+.5f}")

    print("\n  GATE (pre-registered): promote to the 3-seed read iff best_iteration >= 3500")
    if best_iter >= 3500:
        print(f"  -> best_iteration {best_iter} >= 3500: GATE CLEARS, run the paired read.")
    else:
        print(
            f"  -> best_iteration {best_iter} < 3500: GATE FAILS. The config was "
            f"converged; the capacity lever is dead and cost {elapsed / 60:.0f} min."
        )

    # Tail shape, because "peaked late" and "still climbing" are different claims
    # and only the second is capacity. See the module docstring.
    if grown > SHIPPED_CAP:
        tail = curve[SHIPPED_CAP:]
        print(
            f"\n  tail beyond the shipped cap: {len(tail)} rounds, "
            f"range {tail.min():.5f}..{tail.max():.5f} (spread {tail.max() - tail.min():.5f})"
        )

    del model


if __name__ == "__main__":
    main()
