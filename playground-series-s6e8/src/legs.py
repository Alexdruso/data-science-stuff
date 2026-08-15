"""Generic CV runner for a non-LightGBM leg on a chosen feature surface.

`run_arm.py` owns the LightGBM arm sweep; this owns everything else. Both go
through the same `ARMS` table and the same fold seeds, so every OOF array lands
row-aligned and stackable — provided the arm (the *surface*) matches. Arrays
from different arms must never be blended together (root CLAUDE.md lesson #9).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from run_arm import ARMS, SCORES_PATH, assert_budget_constraint, build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

N_FOLDS = 5

#: Per-fold checkpoints, so a hard reboot costs one fold instead of a whole leg.
#: This box power-cuts at random (twice during this campaign): the 08-11 cut
#: killed `xgbt_s1_s42` at fold 4/5 and threw away four completed folds, which
#: is ~37 minutes of XGBoost. `results/` is gitignored, so these never ship.
CKPT_DIR = RESULTS_DIR / "_ckpt"

FitModel = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, int],
    tuple[np.ndarray, np.ndarray],
]


def config_hash(config: object) -> str:
    """A short stable digest of the fit configuration.

    Root CLAUDE.md lesson #12: a cache key must encode every dimension that
    varies. `{name}_{arm}_s{seed}_f{fold}` does not — the capacity experiment
    reuses the `xgbt` machinery with a different `n_estimators`, and without the
    config in the key a stale block would be served straight into the new arm
    and silently fake the result. `default=repr` keeps non-JSON values (numpy
    scalars, callables) hashable rather than raising.
    """
    blob = json.dumps(config, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_leg(
    name: str,
    arm: str,
    seed: int,
    fit_model: FitModel,
    config: object = None,
    resume: bool = True,
) -> float:
    """Fit one leg on one surface at one seed; save OOF/test arrays; return OOF AUC.

    `config` is the fit configuration (typically the params dict) and is hashed
    into the checkpoint key. **Passing nothing disables checkpointing** rather
    than caching under a key that cannot tell two configurations apart — a
    silently-served stale block is far worse than a re-run fold.
    """
    builder, transform, _ = ARMS[arm]
    x, y, x_test = build_matrices(builder)
    assert_budget_constraint(x)

    # Width AFTER the transform, captured from the first fold. Reporting the
    # pre-transform width (as this did until 08-10) cannot distinguish "the
    # transform ran" from "the transform silently no-op'd" — every arm with a
    # transform printed the bare `s1` count of 27 and looked identical to `s1`.
    fitted_width: list[int] = []

    tag = f"{name}_{arm}_s{seed}"
    cfg = config_hash(config) if (resume and config is not None) else None
    if cfg is None:
        print("  checkpointing OFF (no config passed) — a reboot costs the whole leg", flush=True)

    def ckpt_path(fold: int) -> Path:
        return CKPT_DIR / f"{tag}_f{fold}_{cfg}.npz"

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cfg is not None and ckpt_path(fold).exists():
            block = np.load(ckpt_path(fold))
            print(f"    {tag} fold {fold}/{N_FOLDS} restored from checkpoint", flush=True)
            return block["val"], block["test"]
        if transform is not None:
            # Fold-aware: every statistic is fit on the training split only.
            x_tr, x_va, x_te = transform(x_tr, y_tr, x_va, x_te)
            if x_tr.shape[1] == x.shape[1]:
                raise RuntimeError(
                    f"arm {arm}: transform added no columns "
                    f"({x_tr.shape[1]} == {x.shape[1]}) — it is a silent no-op"
                )
        if not fitted_width:
            fitted_width.append(x_tr.shape[1])
        val_pred, test_pred = fit_model(x_tr, y_tr, x_va, y_va, x_te, fold)
        if cfg is not None:
            # Write via a temp file and rename: os.replace is atomic on POSIX, so
            # a power cut can leave the block absent but never half-written. A
            # truncated .npz would be *restored* on the next run, which is the
            # one failure mode a checkpoint must not have.
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path(fold).with_suffix(".tmp.npz")
            np.savez(tmp, val=val_pred, test=test_pred)
            tmp.replace(ckpt_path(fold))
        return val_pred, test_pred

    # A 5-fold leg here runs for tens of minutes and `run_leg` only printed once
    # it was completely done, so a silent log was indistinguishable from a hung
    # or dead run — which is exactly how the 08-06/08-07 sessions lost work.
    started = time.time()

    def report(fold: int) -> None:
        # run_cv enumerates folds from 1, so `fold` is already 1-based.
        print(
            f"    {tag} fold {fold}/{N_FOLDS} done "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )

    oof, test_pred, fold_scores = run_cv(
        x,
        y,
        x_test,
        fit_fold,
        n_splits=N_FOLDS,
        seed=seed,
        n_outputs=1,
        score_fn=lambda yv, p: float(roc_auc_score(yv, p)),
        after_fold=report,
    )
    auc = float(roc_auc_score(y, oof))
    np.save(RESULTS_DIR / f"oof_{tag}.npy", oof)
    np.save(RESULTS_DIR / f"test_{tag}.npy", test_pred)
    save_cv_result(RESULTS_DIR, tag, fold_scores, auc, metric_name="auc")

    scores: dict[str, dict[str, float]] = {}
    if SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text())
    scores.setdefault(f"{name}_{arm}", {})[str(seed)] = auc
    SCORES_PATH.write_text(json.dumps(scores, indent=2, sort_keys=True))

    # Every fold is on disk and the leg is written, so the blocks have no further
    # job. Kept until now rather than deleted per fold: if the *final* np.save is
    # what gets interrupted, the whole run is still one restart away.
    if cfg is not None:
        for fold in range(1, N_FOLDS + 1):
            ckpt_path(fold).unlink(missing_ok=True)

    # `fitted_width` is only populated by folds that actually fit. On a fully
    # restored run nothing fits, so report that rather than printing the surface
    # width as if the transform had been checked — the 08-10 bug this guard
    # exists for was precisely a width that looked verified and was not.
    if fitted_width:
        width = f"{fitted_width[0]:>3}"
    else:
        width = "n/a"
    print(
        f"  {tag}: features={width} (surface {x.shape[1]})  OOF AUC {auc:.5f}",
        flush=True,
    )
    return auc
