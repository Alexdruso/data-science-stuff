"""Optuna re-tune of the te13-recipe HGBC config (Day-12 CPU track).

The Day-5 tuning kill (tune_hgbc.py, +0.0004 flat) ran WITHOUT the te13
features — and the hgbc_te3d ablation proved the config carries ~+0.0007 of
the recipe's +0.0009 (config x features x surface interact; single-axis kills
don't compose). This tuner closes the one unmeasured axis: the Susol config
was Optuna-tuned by HIM on this recipe, so the honest prior is FAIL — but
that FAIL is a measurement, not an assertion.

Objective (cheap, paired): 2 fixed folds (1 and 3 — fold 2 reads ~+0.002 hot)
of the seed-42 zoo_cv split, full rows (no subsampling: it biases exactly the
regularization knobs under test). The te13 blocks come from the existing
results/cache/te13_u_s42_f{1,3}.npz (seeded inner CV; the trainer's inline
encoder is unseeded — fine for RANKING, the winner is confirmed through the
real trainer). Trial value = decision-weighted bacc on the 2-fold partial OOF
(the deployed objective, identical treatment per trial); per-fold argmax bacc
feeds median pruning. Susol's exact config is enqueued as trial 0.

Search space widens around Susol and bridges the Day-5 second basin
(lr~0.2 / depth 7). max_iter is raised 300->600 so low-lr configs aren't
truncated (loss-scored early stopping still ends runs ~100 iters).

Run:  S6E7_TUNE_SECONDS=25200 ../.venv/bin/python src/tune_hgbc_te3.py \
        | tee -a results/tune_hgbc_te3.txt
Rescore top-K on all 5 folds:  .../python src/tune_hgbc_te3.py rescore 3
Study: results/optuna_hgbc_te3.db (NEW db; optuna_hgbc.db is never touched).
Reboot-resumable (sqlite + load_if_exists). Screen bar: best-incumbent >=
+0.0007 here -> full-trainer confirmation (S6E7_HGBC_PARAMS, fresh _u_s42t
tag); PASS = weighted OOF >= incumbent + 0.001 there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, N_FOLDS, RESULTS_DIR, Dataset, load_dataset
from zoo_common import te13_block_for_fold

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SEED = 42
TUNE_FOLDS = [1, 3]  # 1-indexed, matching zoo_cv's enumerate(...); fold 2 reads hot

FIXED: dict[str, object] = {
    # Susol-recipe semantics: default LOSS-scored early stopping (NOT
    # balanced_accuracy — that was the old train_hgbc.py default).
    "early_stopping": True,
    "max_iter": 600,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
    "random_state": SEED,
}

# train_hgbc_te3.HGBC_CONFIG incumbent (Susol v0.7 tuned; weighted OOF 0.9504),
# expressed in the suggest() parameter space. Enqueued as trial 0.
DEFAULT_TRIAL: dict[str, object] = {
    "learning_rate": 0.0627037115235577,
    "max_leaf_nodes": 33,
    "max_depth": 0,  # 0 encodes None
    "min_samples_leaf": 298,
    "l2_regularization": 0.028912644384523085,
    "max_features": 0.820265066682815,
    "max_bins": 237,
}


def suggest(trial: optuna.Trial) -> dict[str, object]:
    depth = trial.suggest_categorical("max_depth", [0, 5, 6, 7, 8, 9])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 96, log=True),
        "max_depth": None if depth == 0 else depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 60, 1200, log=True),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-3, 3.0, log=True
        ),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_bins": trial.suggest_categorical("max_bins", [192, 237, 255]),
    }


def build_fold_designs(
    ds: Dataset, folds: list[int]
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]]:
    """(X_tr, X_va, y_tr, val_idx) per requested fold — raw cols exactly as
    train_hgbc_te3.main (cats as Categorical w/ 'missing', numerics NaN native)
    + the 39 cached te13 columns. Built once; trials only pay the HGBC fit."""
    X = ds.train.copy()
    X_test = ds.test  # only needed for the shared category vocabulary
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        cats = sorted(
            set(X[col].unique()) | set(X_test[col].fillna("missing").astype(str).unique())
        )
        X[col] = pd.Categorical(X[col], categories=cats)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = dict(enumerate(skf.split(ds.y, ds.y), 1))
    designs = {}
    for fold in folds:
        tr_idx, val_idx = splits[fold]
        t13_tr, _t13_te, te_names = te13_block_for_fold(ds, tr_idx, SEED, fold, "u")
        te_names = [str(n) for n in te_names]  # npz round-trip yields np.str_

        def assemble(idx: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [
                    X.iloc[idx].reset_index(drop=True),
                    pd.DataFrame(t13_tr[idx], columns=te_names),
                ],
                axis=1,
            )

        designs[fold] = (assemble(tr_idx), assemble(val_idx), ds.y[tr_idx], val_idx)
        print(f"fold {fold}: design ready ({len(tr_idx)} train / {len(val_idx)} val)")
    return designs


def make_objective(designs, y):  # noqa: ANN001, ANN201
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        val_idx_all: list[np.ndarray] = []
        proba_all: list[np.ndarray] = []
        fold_scores = []
        for step, fold in enumerate(TUNE_FOLDS):
            X_tr, X_va, y_tr, val_idx = designs[fold]
            model = HistGradientBoostingClassifier(**FIXED, **params)
            model.fit(X_tr, y_tr)
            assert list(model.classes_) == list(range(N_CLASSES))
            proba = model.predict_proba(X_va)
            val_idx_all.append(val_idx)
            proba_all.append(proba)
            fold_scores.append(
                balanced_accuracy_score(y[val_idx], proba.argmax(axis=1))
            )
            trial.set_user_attr(f"n_iter_f{fold}", int(model.n_iter_))
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        idx = np.concatenate(val_idx_all)
        proba = np.concatenate(proba_all)
        w = tune_decision_weights(y[idx], proba, n_restarts=3)
        return float(balanced_accuracy_score(y[idx], weighted_predict(proba, w)))

    return objective


def rescore_top(k: int) -> None:
    """Confirmation rung 1: rescore the study's top-k distinct configs (plus the
    incumbent) on ALL 5 folds with the same dw objective — kills 2-fold luck."""
    study = optuna.load_study(
        study_name="hgbc_te3",
        storage=f"sqlite:///{RESULTS_DIR / 'optuna_hgbc_te3.db'}",
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(done, key=lambda t: -(t.value or 0.0))
    picks = [t for t in ranked[:k] if t.params != DEFAULT_TRIAL]
    incumbent = next(t for t in done if t.params == DEFAULT_TRIAL)

    ds = load_dataset()
    all_folds = list(range(1, N_FOLDS + 1))
    designs = build_fold_designs(ds, all_folds)
    y = ds.y
    for t in [incumbent, *picks]:
        params = dict(t.params)
        depth = params.pop("max_depth")
        params["max_depth"] = None if depth == 0 else depth
        oof = np.zeros((len(y), N_CLASSES))
        for fold in all_folds:
            X_tr, X_va, y_tr, val_idx = designs[fold]
            model = HistGradientBoostingClassifier(**FIXED, **params)
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_va)
        w = tune_decision_weights(y, oof, n_restarts=3)
        score = balanced_accuracy_score(y, weighted_predict(oof, w))
        label = "incumbent" if t.number == 0 else f"trial {t.number}"
        print(f"5-fold dw bacc [{label}]: {score:.5f}  params={t.params}")


def main() -> None:
    ds = load_dataset()
    designs = build_fold_designs(ds, TUNE_FOLDS)
    objective = make_objective(designs, ds.y)

    study = optuna.create_study(
        study_name="hgbc_te3",
        storage=f"sqlite:///{RESULTS_DIR / 'optuna_hgbc_te3.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    # pin the incumbent unless a trial with its exact params already exists
    # (params-match, not trial-number: a crash mid-trial-0 must re-pin it)
    if not any(t.params == DEFAULT_TRIAL for t in study.trials):
        study.enqueue_trial(DEFAULT_TRIAL)

    timeout = int(os.environ.get("S6E7_TUNE_SECONDS", "3600"))
    study.optimize(objective, timeout=timeout, gc_after_trial=True)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{len(done)} complete trials ({len(study.trials)} total)")
    incumbent = next((t for t in done if t.params == DEFAULT_TRIAL), None)
    base = incumbent.value if incumbent else float("nan")
    print(f"incumbent (Susol v0.7 config, 2-fold dw): {base:.5f}")
    print(f"best: {study.best_value:.5f} (trial {study.best_trial.number})")
    print(f"best params: {study.best_params}")
    delta = study.best_value - base
    print(
        f"headroom vs incumbent: {delta:+.5f} -> "
        + (
            "SCREEN PASS (rescore top-3 on 5 folds, then full-trainer confirm "
            "via S6E7_HGBC_PARAMS + S6E7_RUN_TAG=_u_s42t)"
            if delta >= 0.0007
            else "FAIL (config x te13 axis closed by measurement)"
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rescore":
        rescore_top(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        main()
