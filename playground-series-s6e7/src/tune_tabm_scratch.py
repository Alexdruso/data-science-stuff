"""Optuna sweep of the from-scratch TabM config (Day-12 GPU filler).

Explicitly a filler lane (near-zero EV: tabms 0.9490 solo on the repaired
surface, no LR seat) run only because the RealMLP ladder finished early and
the GPU would otherwise idle. Hand-set constants (LR/WD/DROPOUT/TAU/HIDDEN)
were never swept.

Objective mirrors tune_hgbc_te3.py: 2 fixed folds (1 and 3) of the seed-42
zoo_cv split on the UNREPAIRED surface (te_u_s42_f* caches hit), trial value
= decision-weighted bacc on the 2-fold partial OOF, per-fold argmax bacc for
median pruning. The incumbent constants are enqueued as trial 0 (also the
first u-surface tabms read — the 0.9490 reference is repaired-surface).

Reuses train_tabm_scratch.fit_fold_impl by monkey-patching the module
constants per trial (classes read them at instantiation time).

Run:  ../.venv/bin/python src/tune_tabm_scratch.py | tee -a results/tune_tabm_scratch.txt
Env:  S6E7_TUNE_SECONDS (default 7200).
Study: results/optuna_tabms.db. Reboot-resumable. Killable at any time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import train_tabm_scratch as tts
from train_common import N_FOLDS, RESULTS_DIR, load_dataset
from train_mlp import NNData

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SEED = 42
TUNE_FOLDS = [1, 3]

# module-constant incumbent (train_tabm_scratch.py hand-set values)
DEFAULT_TRIAL: dict[str, object] = {
    "lr": 2e-3,
    "wd": 1e-5,
    "dropout": 0.15,
    "tau": 1.075,
    "hidden": 512,
}


def suggest(trial: optuna.Trial) -> dict[str, object]:
    return {
        "lr": trial.suggest_float("lr", 4e-4, 8e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "tau": trial.suggest_float("tau", 0.9, 1.3),
        "hidden": trial.suggest_categorical("hidden", [256, 512, 768]),
    }


def apply_params(params: dict[str, object]) -> None:
    tts.LR = float(params["lr"])  # type: ignore[assignment]
    tts.WD = float(params["wd"])  # type: ignore[assignment]
    tts.DROPOUT = float(params["dropout"])  # type: ignore[assignment]
    tts.TAU = float(params["tau"])  # type: ignore[assignment]
    tts.HIDDEN = int(params["hidden"])  # type: ignore[assignment]


def main() -> None:
    assert tts.SURFACE_TAG == "u", "run without S6E7_REPAIR (u surface)"
    ds = load_dataset()
    nd = NNData(ds)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = dict(enumerate(skf.split(ds.y, ds.y), 1))

    def objective(trial: optuna.Trial) -> float:
        import torch

        params = suggest(trial)
        apply_params(params)
        val_idx_all, proba_all, fold_scores = [], [], []
        for step, fold in enumerate(TUNE_FOLDS):
            tr_idx, val_idx = splits[fold]
            try:
                proba, _ = tts.fit_fold_impl(
                    ds, nd, tr_idx, val_idx, SEED, fold, want_test=False
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise optuna.TrialPruned from None
            val_idx_all.append(val_idx)
            proba_all.append(proba)
            fold_scores.append(
                balanced_accuracy_score(ds.y[val_idx], proba.argmax(axis=1))
            )
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        idx = np.concatenate(val_idx_all)
        proba = np.concatenate(proba_all)
        w = tune_decision_weights(ds.y[idx], proba, n_restarts=3)
        return float(
            balanced_accuracy_score(ds.y[idx], weighted_predict(proba, w))
        )

    study = optuna.create_study(
        study_name="tabms",
        storage=f"sqlite:///{RESULTS_DIR / 'optuna_tabms.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0),
    )
    if not any(t.params == DEFAULT_TRIAL for t in study.trials):
        study.enqueue_trial(DEFAULT_TRIAL)

    timeout = int(os.environ.get("S6E7_TUNE_SECONDS", "7200"))
    study.optimize(objective, timeout=timeout, gc_after_trial=True)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{len(done)} complete trials ({len(study.trials)} total)")
    incumbent = next((t for t in done if t.params == DEFAULT_TRIAL), None)
    base = incumbent.value if incumbent else float("nan")
    print(f"incumbent (hand-set constants, 2-fold dw, u surface): {base:.5f}")
    print(f"best: {study.best_value:.5f} (trial {study.best_trial.number})")
    print(f"best params: {study.best_params}")
    delta = study.best_value - base
    print(
        f"headroom vs incumbent: {delta:+.5f} -> "
        + ("SCREEN PASS (confirm via full 5-fold)" if delta >= 0.0007
           else "FAIL (TabM-scratch config axis closed by measurement)")
    )


if __name__ == "__main__":
    main()
