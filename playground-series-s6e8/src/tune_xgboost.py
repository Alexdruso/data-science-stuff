"""Optuna hyperparameter search for the XGBoost leg.

    python src/tune_xgboost.py --arm s1 --trials 60 --timeout 14400

**Why XGBoost and not LightGBM.** The Day-10 breadth blend puts ~0.78 of its
weight on XGBoost and ~0.22 on the six-seed LightGBM bag; CatBoost and the MLP
both get exactly 0.0. Tuning effort should follow the weight. `PARAMS` in
`train_xgboost.py` are hand-set (lr 0.03 / depth 8 / 3000 trees) and have never
been searched.

Tuning is also the one lever a combiner cannot absorb. Root CLAUDE.md lesson #7
says a strong combiner eats leg-level gains — today's CatBoost repair was worth
+0.00272 solo and **+0.000000** to the blend, which is that lesson in one line.
But tuning *replaces* the leg rather than adding a member, so there is no
combiner above it to absorb the gain. That is why the pre-registration gives
tuning the lower +0.0005 bar.

Mirrors `tune_lgbm.py` deliberately:

* **3 folds, not 5** — hyperparameter *ranking* is a coarse decision; the winner
  is re-fit at 5 folds by `train_xgboost.py`.
* **sqlite-backed and resumable.** This box hard-cut power mid-run on 2026-08-06
  at 09:40:53 and lost a morning. `load_if_exists` makes the identical command a
  resume, so a crash costs exactly the running trial.
* **The untuned reference is enqueued as trial 0** on a fresh study, so every
  later trial is read against the incumbent rather than against nothing. On the
  LightGBM study that reference is *still* the best after six trials — a seeded
  reference is what makes that fact visible instead of invisible.

Device is CPU to match the leg it is tuning (`train_xgboost.py` uses
`tree_method="hist"` on CPU); a device switch would move the number by about the
size of the effect being measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from run_arm import ARMS, build_matrices

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

TUNE_FOLDS = 3

#: Bumped whenever the search space changes, so a widened/narrowed space starts a
#: NEW study rather than feeding TPE a history sampled from different bounds.
#: The old study stays on disk as the record of what was covered.
#: v1 -> v2 (2026-08-10): raised the reg_alpha/reg_lambda/gamma floors off 1e-8
#: and dropped max_bin=128. **This changes the search space, not the bar** — the
#: pre-registered adoption threshold is still +0.0005 on a 3-seed paired 5-fold.
#: v3 -> v4 (2026-08-15): the adopted point sat ON two of its own bounds, so the
#: search was truncated there rather than converged. Both are opened:
#:   * `max_bin` 512 was the TOP of {256, 512} -> {256, 512, 1024}. Two independent
#:     findings say fine binning pays on this data (CatBoost border_count 128->254
#:     was +0.00285 solo; every v1 trial drawing max_bin=128 landed among the worst),
#:     and the signal lives in fine-grained continuous columns.
#:   * `learning_rate` floor 0.015 was never breached -> 0.005. A lower lr is a
#:     question about STEP SIZE, which the 08-15 capacity read (n_estimators, clean
#:     negative at +0.00007) explicitly does not answer.
#: `n_estimators` is raised to 6000 as an ENABLER, not an arm: at lr 0.005 the
#: 3000 cap would truncate the search all over again, and the capacity read showed
#: a longer leash is worth ~0 on its own at a fixed lr.
#: **The bar does not move: +0.0005 on a 3-seed paired 5-fold, same as 08-13.**
SPACE_VERSION = "v4"
FIXED: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    # Raised from 3000 with the v4 space: at the new lr floor of 0.005 a 3000 cap
    # binds long before convergence, which is the exact truncation v4 exists to
    # remove. Early stopping still decides the real count, and the 08-15 capacity
    # read established that a longer leash alone buys ~0 (+0.00007, 0.14x the bar).
    "n_estimators": 6000,
    "early_stopping_rounds": 100,
    "enable_categorical": True,
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": 42,
}

#: The shipped `train_xgboost.py` point, seeded as trial 0 on a fresh study.
REFERENCE = {
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    # The shipped leg sets neither, so XGBoost defaults both to 0. 1e-3 is the
    # v2 floor and is the nearest in-range stand-in — numerically a no-op for a
    # min-split-loss and an L1 term at this scale. Enqueueing the literal 1e-8
    # made Optuna warn that the fixed value was out of range. This only affects
    # the reference point *inside the search*; adoption is decided by a 5-fold
    # paired refit against the real shipped PARAMS, not against this trial.
    "reg_alpha": 1e-3,
    "gamma": 1e-3,
    "max_bin": 256,
}

#: Known-good points carried across a SPACE_VERSION bump so their information is
#: not thrown away with the old study. The first is the only point measured so far
#: that beats the reference: 0.96463 vs 0.96409 at 3 folds, found in the
#: direction-broken v2 study before it was discarded.
SEED_POINTS: list[dict[str, object]] = [
    # v3 trial 8 — the ADOPTED point, 3-fold 0.96530, and the one that cleared the
    # +0.0005 bar at +0.001139 on the 5-fold paired read. Enqueued first so a wider
    # v4 box cannot lose ground v3 already holds: TPE starts from a known optimum
    # instead of rediscovering it. Note it will re-evaluate at n_estimators=6000
    # rather than 3000, which the capacity read priced at +0.00007 — so its v4 value
    # should land within noise of 0.96530 and is directly comparable.
    {
        "learning_rate": 0.028751410425172524,
        "max_depth": 7,
        "min_child_weight": 16.755052359850303,
        "subsample": 0.7427013306774357,
        "colsample_bytree": 0.5685607058124285,
        "reg_lambda": 1.2915099897772668,
        "reg_alpha": 0.0036618192203924276,
        "gamma": 0.9274757630456326,
        "max_bin": 512,
    },
    # The same point with max_bin pushed to the new 1024 option and nothing else
    # changed — the one-factor `max_bin` read, evaluated as trial 2 instead of as a
    # separate hand-run arm. If finer binning pays, this is where it shows first.
    {
        "learning_rate": 0.028751410425172524,
        "max_depth": 7,
        "min_child_weight": 16.755052359850303,
        "subsample": 0.7427013306774357,
        "colsample_bytree": 0.5685607058124285,
        "reg_lambda": 1.2915099897772668,
        "reg_alpha": 0.0036618192203924276,
        "gamma": 0.9274757630456326,
        "max_bin": 1024,
    },
    # The v2 point that first beat the reference (0.96463 vs 0.96409 at 3 folds),
    # carried forward across every space bump. A deep/low-regularisation optimum,
    # structurally different from trial 8's shallow/heavily-regularised one.
    {
        "learning_rate": 0.028078987855609694,
        "max_depth": 12,
        "min_child_weight": 12.057126287443763,
        "subsample": 0.8394633936788146,
        "colsample_bytree": 0.4936111842654619,
        "reg_lambda": 0.12731427596811182,
        "reg_alpha": 0.0017073967431528124,
        "gamma": 1.599409170128044,
        "max_bin": 512,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--out", default="best_params_xgboost.json")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    FIXED["n_jobs"] = args.n_jobs

    builder, _, description = ARMS[args.arm]
    x, y, _ = build_matrices(builder)
    print(f"tuning xgb on arm {args.arm} ({description}): {x.shape}", flush=True)

    folds = list(
        StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=42).split(x, y)
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED,
            # Floor 0.015 -> 0.005 for v4: no v3 trial ever went below 0.015, so the
            # region was never sampled rather than sampled and rejected.
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # Floors raised off 1e-8. A log-uniform [1e-8, 20] puts ~85% of its
            # mass below 0.01, where these are effectively OFF — so v1 sampled
            # "no regularisation" almost every trial while the incumbent's only
            # distinguishing setting was reg_lambda=1.0. Measured over v1's 9
            # trials: best had lambda 1.000, every other had <=0.002.
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            # 128 dropped: all three v1 trials that drew it landed among the four
            # worst. Independently consistent with the CatBoost finding that
            # border_count=128 costs real score here — this dataset's signal is
            # in fine-grained continuous columns, so coarse binning hurts.
            # 1024 added in v4: the adopted point chose 512, the TOP of the old set,
            # so "finer is better" was still improving when the options ran out.
            "max_bin": trial.suggest_categorical("max_bin", [256, 512, 1024]),
        }
        oof = np.zeros(len(y))
        best_iters: list[int] = []
        for tr_idx, va_idx in folds:
            model = XGBClassifier(**params)
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(x.iloc[va_idx], y[va_idx])],
                verbose=False,
            )
            oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
            best_iters.append(int(model.best_iteration))
            del model
        # The MAX across folds, not the last fold's. Overwriting per fold (as this
        # did until 08-15) discards the only signal that says whether n_estimators
        # is binding — which is exactly how the adopted v3 point came to sit at
        # best_iteration 2999/3000 without anyone noticing until Day 15. At v4's
        # lr floor of 0.005 the 6000 cap is the one to watch.
        trial.set_user_attr("best_iteration", max(best_iters))
        trial.set_user_attr("best_iteration_per_fold", best_iters)
        return float(roc_auc_score(y, oof))

    db = RESULTS_DIR / f"optuna_xgb_{args.arm}.db"

    # The sampler seed must ADVANCE with the study, not be pinned. TPE samples
    # its first `n_startup_trials` (default 10) at random, and the sampler is
    # rebuilt on every launch — so a fixed seed makes each resume replay the same
    # random draws instead of continuing the search. Observed live on 08-10:
    # after a restart, trials 6 and 7 reproduced trials 2 and 3 to the digit
    # (0.96385, 0.95985), burning ~4 min each to re-learn nothing.
    # Offsetting by the completed-trial count keeps a from-scratch run exactly
    # reproducible while making a resume continue rather than repeat.
    # `load_study`, never `create_study`, for the probe. `create_study` without an
    # explicit `direction` creates a **minimize** study, and a later
    # `create_study(direction="maximize", load_if_exists=True)` silently returns
    # that existing mis-directed study rather than correcting it. That happened on
    # 2026-08-10: study `xgb_s1_v2` minimised AUC, so a trial scoring 0.96463 lost
    # to one scoring 0.96409 and TPE would have optimised toward worse params all
    # evening. `load_study` raises instead of creating, so the direction is set in
    # exactly one place.
    n_done = 0
    if db.exists():
        try:
            probe = optuna.load_study(
                storage=f"sqlite:///{db}", study_name=f"xgb_{args.arm}_{SPACE_VERSION}"
            )
        except KeyError:
            probe = None
        if probe is not None:
            n_done = len(
                [t for t in probe.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            if probe.direction is not optuna.study.StudyDirection.MAXIMIZE:
                msg = (
                    f"study 'xgb_{args.arm}_{SPACE_VERSION}' is "
                    f"{probe.direction.name}, not MAXIMIZE — refusing to resume it. "
                    "Bump SPACE_VERSION or delete the study."
                )
                raise RuntimeError(msg)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42 + n_done),
        storage=f"sqlite:///{db}",
        study_name=f"xgb_{args.arm}_{SPACE_VERSION}",
        load_if_exists=True,
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        print(
            f"resuming study 'xgb_{args.arm}' from {db.name}: "
            f"{len(done)} completed trials, best so far {study.best_value:.5f}",
            flush=True,
        )
    if not study.trials:
        study.enqueue_trial(dict(REFERENCE))
        for extra in SEED_POINTS:
            study.enqueue_trial(dict(extra))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out = RESULTS_DIR / args.out

    def write_best(study: optuna.Study) -> None:
        best = {**FIXED, **study.best_params}
        best["n_estimators"] = 3000
        out.write_text(json.dumps(best, indent=2, sort_keys=True))

    # Write immediately on resume, not only on improvement. Otherwise a resumed
    # study whose incumbent is never beaten — the likely case when the enqueued
    # reference is strong — keeps its best params in sqlite and never puts them
    # on disk, which is the exact failure the incremental write exists to avoid.
    if done:
        write_best(study)

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Write on every improvement, not once at the end. `study.optimize` only
        # returns when trials/timeout are exhausted, so a search that is killed
        # early — or cut by this box's habit of losing power mid-run — would
        # otherwise leave a database full of results and no usable params file.
        # Same reasoning as the sqlite storage: a crash should cost one trial.
        if trial.value is not None and trial.value >= study.best_value:
            write_best(study)
        print(
            f"  trial {trial.number:>3}  {trial.value:.5f}   best {study.best_value:.5f}",
            flush=True,
        )

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=[log])

    write_best(study)
    print(f"\nbest {TUNE_FOLDS}-fold OOF AUC: {study.best_value:.5f}")
    print(f"best params -> {out}")
    print(json.dumps(study.best_params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
