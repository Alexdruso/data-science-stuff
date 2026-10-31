"""The Optuna study scaffold every ``tune_<model>.py`` repeats.

19 tune scripts across s6e5-s6e10 each rebuilt the same pieces: a seeded TPE
sampler, an optional resumable SQLite study, the default config enqueued as
trial 0, ``gc_after_trial`` plus ``torch.cuda.empty_cache()`` between trials,
and a flat ``best_params*.json`` that the matching ``train_<model>.py`` reads
back with :func:`data_science_stuff.kaggle.io.load_params`. :func:`run_study`
owns those pieces; the script keeps only its objective.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import optuna


def cuda_cleanup_callback(
    _study: optuna.Study, _trial: optuna.trial.FrozenTrial
) -> None:
    """Optuna callback: free cached CUDA memory after every trial.

    The GPU memory rule: without this a multi-trial GPU search fragments the
    allocator and OOM-crashes partway through. A no-op without CUDA.
    """
    gc.collect()
    try:
        import torch  # noqa: PLC0415 — torch is optional at call time
    except ImportError:  # pragma: no cover — torch is a declared dependency
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def best_params_writer(
    path: Path, fixed: Mapping[str, Any] | None = None
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Optuna callback that rewrites ``path`` whenever the best trial improves.

    Written on every improvement rather than once at the end, so a search
    killed mid-run (timeouts, reboots) still leaves its best params on disk.
    The file is ``{**fixed, **best_params}`` — flat JSON in the shape
    :func:`data_science_stuff.kaggle.io.load_params` merges over defaults.
    """
    fixed_params = dict(fixed or {})

    def _write(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        if not completed or study.best_trial.number != trial.number:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**fixed_params, **study.best_params}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return _write


def run_study(
    objective: Callable[[optuna.Trial], float],
    *,
    n_trials: int | None = None,
    timeout: float | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
    study_name: str | None = None,
    storage_path: Path | None = None,
    seed: int = 42,
    enqueue: Sequence[Mapping[str, Any]] = (),
    params_path: Path | None = None,
    fixed: Mapping[str, Any] | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    cuda_cleanup: bool = True,
    callbacks: Sequence[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = (),
) -> optuna.Study:
    """Create (or resume) a study, run ``objective``, and persist the best params.

    Args:
        objective: ``trial -> score``. Put the CV loop here; its fold loop must
            still call ``torch.cuda.empty_cache()`` per fold for GPU models.
        n_trials: Trial budget for this call (``None`` = until ``timeout``).
        timeout: Wall-clock budget in seconds.
        direction: ``"maximize"`` or ``"minimize"``.
        study_name: Required with ``storage_path`` to resume a study.
        storage_path: SQLite file; when given the study is resumable
            (``load_if_exists=True``) across invocations.
        seed: TPE sampler seed.
        enqueue: Param dicts to try first (typically the hand-tuned defaults,
            so the search can only beat them). Skipped when resuming a study
            that already has trials.
        params_path: Where :func:`best_params_writer` keeps the best params,
            e.g. ``results/best_params_xgboost.json``.
        fixed: Non-searched params merged under the searched ones in
            ``params_path``.
        pruner: Optional pruner (e.g. ``optuna.pruners.MedianPruner()``).
        cuda_cleanup: Register :func:`cuda_cleanup_callback`.
        callbacks: Extra Optuna callbacks.

    Returns:
        The study (``study.best_params``, ``study.trials_dataframe()``, ...).
    """
    if storage_path is not None and study_name is None:
        msg = "study_name is required to resume a study from storage_path"
        raise ValueError(msg)
    storage = f"sqlite:///{storage_path}" if storage_path is not None else None
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
    )
    if not study.trials:
        for params in enqueue:
            study.enqueue_trial(dict(params))

    all_callbacks = list(callbacks)
    if params_path is not None:
        all_callbacks.append(best_params_writer(params_path, fixed))
    if cuda_cleanup:
        all_callbacks.append(cuda_cleanup_callback)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        callbacks=all_callbacks,
    )
    return study
