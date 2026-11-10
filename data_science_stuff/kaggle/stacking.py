"""Stacking: honest multi-seed OOF meta-learning and Caruana greedy selection.

A scalar blend (see :mod:`data_science_stuff.kaggle.blending`) assigns one
weight per model, so a model that is strong on one class and weak on another
gets ~0 weight — no scalar captures "use its class-A column, ignore its
class-B column". :func:`stack_oof` fits a meta-learner on the base models'
per-class outputs instead (a weight per model-class pair with a linear meta),
which is what broke s6e6's 0.9662 plateau (LR-on-logits stack → 0.9705).

Honesty rules baked in:

- The meta-learner is fit on K-1 folds of the OOF rows and predicts the
  held-out fold — never fit and scored on the same rows.
- :func:`caruana_select` reports held-out scores from an outer CV around the
  greedy selection; naive best-on-OOF selection manufactures a fake gain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_science_stuff.kaggle.blending import ScoreFn


def clipped_logit(p: NDArray[np.float64], *, eps: float = 1e-6) -> NDArray[np.float64]:
    """Elementwise ``log(p)`` with probabilities clipped away from {0, 1}.

    Stacking on logits instead of raw probabilities lets a linear meta-learner
    operate on an unbounded, roughly Gaussian scale.

    >>> import numpy as np
    >>> clipped_logit(np.array([0.5])).round(4).tolist()
    [-0.6931]
    """
    return np.asarray(np.log(np.clip(p, eps, 1 - eps)), dtype=np.float64)


def logit(p: NDArray[Any], *, eps: float = 1e-15) -> NDArray[np.float64]:
    """Binary log-odds ``log(p / (1 - p))`` with ``p`` clipped to ``[eps, 1 - eps]``.

    Not to be confused with :func:`clipped_logit`, which is the multiclass
    ``log(p)``. The default ``eps`` only guards exact 0/1 probabilities; it is
    the value the s6e11 scripts had re-written locally.

    >>> import numpy as np
    >>> logit(np.array([0.5, 0.75])).round(4).tolist()
    [0.0, 1.0986]
    """
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def rank_logit(v: NDArray[Any]) -> NDArray[np.float64]:
    """Logit of the (average) rank of ``v``, scaled into ``(0, 1)`` as ``r / (n + 1)``.

    Puts every stack member on one scale whatever it outputs (probabilities,
    raw margins, rank means). On s6e11's public OOF pool this beat
    standardized logits as LR stacker input (0.952304 vs 0.952230), and
    re-ranking within each fold instead of globally was null.

    >>> import numpy as np
    >>> rank_logit(np.array([3.0, 1.0, 2.0])).round(4).tolist()
    [1.0986, -1.0986, 0.0]
    """
    r = rankdata(v) / (len(v) + 1)
    return np.asarray(np.log(r / (1 - r)), dtype=np.float64)


def _as_columns(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def stack_oof(
    base_oofs: Sequence[NDArray[np.float64]],
    base_tests: Sequence[NDArray[np.float64]],
    y: NDArray[np.int64],
    meta_factory: Callable[[], Any],
    *,
    seeds: Sequence[int] = (2024, 7, 13, 42, 99),
    n_splits: int = 5,
    use_logits: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Honest multi-seed OOF stacking over base-model prediction arrays.

    The feature matrix is the horizontal stack of every base model's OOF
    array (optionally logit-transformed). For each seed, a fresh meta-learner
    is fit on K-1 folds of those rows and predicts the held-out fold; test
    predictions are averaged over every (seed, fold) fit. Multi-seed
    averaging smooths the fold-boundary noise of any single split.

    Prune the base list to strong, *diverse* models first: near-duplicate
    variants add only collinearity (on s6e6, six weak bases added +0.0001).
    If the stack is flat, the missing ingredient is a diverse base — a
    different feature space, loss, or decomposition — not a better meta.

    Args:
        base_oofs: One OOF array per base model, aligned with ``y``
            ((n, n_classes) probabilities, or 1-D for binary/regression).
        base_tests: Matching test arrays, same order and widths.
        y: Encoded training labels (drives the stratified folds).
        meta_factory: Zero-arg callable returning a *fresh* estimator with
            ``fit`` / ``predict_proba`` (e.g. ``lambda: LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000)``).
        seeds: One outer CV per seed; predictions averaged across them.
        n_splits: Folds per seed.
        use_logits: Feed :func:`clipped_logit` of the arrays to the meta
            (recommended for linear metas; use False for tree metas).

    Returns:
        ``(oof, test)`` stacked probability arrays, shape (n, n_classes) /
        (n_test, n_classes).
    """
    if len(base_oofs) != len(base_tests):
        msg = "base_oofs and base_tests must list the same models"
        raise ValueError(msg)
    if not base_oofs:
        msg = "need at least one base model to stack"
        raise ValueError(msg)

    transform = clipped_logit if use_logits else _as_columns
    z_train = np.hstack([transform(_as_columns(a)) for a in base_oofs])
    z_test = np.hstack([transform(_as_columns(a)) for a in base_tests])

    n_classes = int(np.unique(y).size)
    oof = np.zeros((len(y), n_classes))
    test = np.zeros((len(z_test), n_classes))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(z_train, y):
            meta = meta_factory()
            meta.fit(z_train[tr_idx], y[tr_idx])
            oof[va_idx] += meta.predict_proba(z_train[va_idx]) / len(seeds)
            test += meta.predict_proba(z_test) / (len(seeds) * n_splits)
    return oof, test


class CaruanaResult(NamedTuple):
    """Held-out scores per outer fold, plus the full-data greedy composition."""

    holdout_scores: list[float]
    full_picks: list[int]
    full_score: float


def _greedy_select(
    library: NDArray[np.float64],
    y: NDArray[np.int64],
    idx: NDArray[Any],
    *,
    n_steps: int,
    init_topk: int,
    score: ScoreFn,
) -> list[int]:
    """Greedy forward selection with replacement on the given rows."""
    n_models = library.shape[0]
    singles = [float(score(y[idx], library[m][idx])) for m in range(n_models)]
    picks = [int(m) for m in np.argsort(singles)[::-1][:init_topk]]
    running = library[picks][:, idx, :].sum(axis=0)
    for _ in range(n_steps - len(picks)):
        best_m, best_score = -1, -np.inf
        for m in range(n_models):
            candidate_score = float(score(y[idx], running + library[m][idx]))
            if candidate_score > best_score:
                best_score, best_m = candidate_score, m
        picks.append(best_m)
        running = running + library[best_m][idx]
    return picks


def caruana_select(
    oofs: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    n_steps: int = 50,
    init_topk: int = 3,
    n_outer: int = 5,
    seed: int = 42,
    score_fn: ScoreFn | None = None,
) -> CaruanaResult:
    """Caruana greedy forward selection (with replacement), honestly gated.

    Averages probabilities of greedily-picked base models (a model picked
    twice counts twice). Selection runs inside an outer CV so
    ``holdout_scores`` is an honest estimate — compare its mean against the
    incumbent combiner before trusting the full-data composition. Exclude
    meta/stack arrays from ``oofs`` (selecting a stack is circular).

    Args:
        oofs: (n_models, n_rows, n_classes) library of base OOF arrays.
        y: Encoded training labels.
        n_steps: Total greedy picks (with replacement).
        init_topk: Seed the ensemble with the K best singletons.
        n_outer: Outer honest-CV folds.
        seed: Outer CV ``random_state``.
        score_fn: ``(y_true, summed_probs) -> float``; defaults to balanced
            accuracy on argmax.

    Returns:
        :class:`CaruanaResult` — held-out fold scores, the (optimistic)
        full-data pick list, and its in-sample score.
    """
    score: ScoreFn = (
        score_fn
        if score_fn is not None
        else lambda yt, p: float(balanced_accuracy_score(yt, np.argmax(p, axis=1)))
    )
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    holdout_scores: list[float] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        picks = _greedy_select(
            oofs, y, tr_idx, n_steps=n_steps, init_topk=init_topk, score=score
        )
        held_out = oofs[picks][:, va_idx, :].sum(axis=0)
        holdout_scores.append(float(score(y[va_idx], held_out)))

    full_picks = _greedy_select(
        oofs, y, np.arange(len(y)), n_steps=n_steps, init_topk=init_topk, score=score
    )
    full_score = float(score(y, oofs[full_picks].sum(axis=0)))
    return CaruanaResult(holdout_scores, full_picks, full_score)


def ridge_calibrate(
    per_base_tr: Sequence[NDArray[np.float64]],
    eval_sets: Sequence[Sequence[NDArray[np.float64]]],
    y_tr: NDArray[np.int64],
    *,
    alpha: float = 10.0,
    n_classes: int | None = None,
    balanced: bool = True,
) -> list[NDArray[np.float64]]:
    """Per-base ridge map from a model's outputs to one-hot targets.

    Each base model's (n, k) block (logits or probabilities) is standardised
    and mapped to the one-hot target by a ridge fit **on the training split
    only**; the calibrated blocks are then hstacked as meta-features. This
    puts differently-calibrated bases on a common scale before a meta-learner
    (s6e6/s6e7 ``build_alt_meta.py``). Call it inside the meta-learner's fold
    loop: fitting on the rows being scored leaks.

    Args:
        per_base_tr: One (n_train, k) block per base model.
        eval_sets: Further row sets to transform with the same fitted maps,
            each a list aligned with ``per_base_tr`` (e.g. ``[val_blocks,
            test_blocks]``).
        y_tr: Integer labels of the training rows.
        alpha: Ridge strength.
        n_classes: Defaults to ``y_tr.max() + 1``.
        balanced: Weight rows by inverse class frequency (sklearn
            "balanced") so minority classes are not ignored.

    Returns:
        ``[X_tr, X_eval_1, ...]``: one hstacked (n, n_bases * n_classes)
        matrix per input row set.
    """
    k = int(n_classes if n_classes is not None else y_tr.max() + 1)
    target = np.eye(k)[y_tr]
    weights = None
    if balanced:
        counts = np.maximum(np.bincount(y_tr, minlength=k), 1).astype(float)
        weights = (len(y_tr) / (k * counts))[y_tr]
    for es in eval_sets:
        if len(es) != len(per_base_tr):
            msg = (
                f"every eval set needs one block per base: got {len(es)}, "
                f"expected {len(per_base_tr)}"
            )
            raise ValueError(msg)
    outs: list[list[NDArray[np.float64]]] = [[] for _ in range(1 + len(eval_sets))]
    for i, block in enumerate(per_base_tr):
        scaler = StandardScaler().fit(_as_columns(block))
        ridge = Ridge(alpha=alpha).fit(
            scaler.transform(_as_columns(block)), target, sample_weight=weights
        )
        outs[0].append(ridge.predict(scaler.transform(_as_columns(block))))
        for j, es in enumerate(eval_sets, 1):
            outs[j].append(ridge.predict(scaler.transform(_as_columns(es[i]))))
    return [np.hstack(o) for o in outs]


class ExternalPredictions(NamedTuple):
    """OOF/test pairs read from external (e.g. public-notebook) output dirs."""

    names: list[str]
    oof: list[NDArray[np.float64]]
    test: list[NDArray[np.float64]]
    skipped: dict[str, str]


def load_public_predictions(
    dirs: Sequence[Path],
    *,
    n_train: int,
    n_test: int,
    oof_glob: str = "oof_*.npy",
) -> ExternalPredictions:
    """Collect ``oof_<x>.npy``/``test_<x>.npy`` pairs from downloaded notebook outputs.

    Extracted from s6e10's ``stack_public.load_pairs``. Every pair is checked
    before use and rejected (with the reason in ``skipped``) when its test
    file is missing, its row counts don't match ``n_train``/``n_test``, the
    two arrays disagree on output width, or it contains NaN/inf.

    **Row order is not checked and cannot be:** external arrays are in their
    author's row order (usually sorted by ``id``), not necessarily
    ``build_features()`` order. Confirm both orders agree before stacking —
    a misaligned pair still has the right shape and silently scores ~0.5.

    Args:
        dirs: Directories, one per external source (``kaggle kernels output``).
        n_train: Expected OOF rows.
        n_test: Expected test rows.
        oof_glob: Pattern for the OOF files; the test file name replaces the
            leading ``oof_`` with ``test_``.

    Returns:
        :class:`ExternalPredictions`; names are ``<dir name>/<stem>``.
    """
    names: list[str] = []
    oofs: list[NDArray[np.float64]] = []
    tests: list[NDArray[np.float64]] = []
    skipped: dict[str, str] = {}
    for d in dirs:
        for oof_path in sorted(Path(d).glob(oof_glob)):
            name = f"{Path(d).name}/{oof_path.stem.removeprefix('oof_')}"
            test_path = oof_path.with_name(oof_path.name.replace("oof_", "test_", 1))
            if not test_path.exists():
                skipped[name] = "no matching test file"
                continue
            oof = np.asarray(np.load(oof_path), dtype=np.float64)
            test = np.asarray(np.load(test_path), dtype=np.float64)
            if len(oof) != n_train or len(test) != n_test:
                skipped[name] = (
                    f"rows {len(oof)}/{len(test)}, expected {n_train}/{n_test}"
                )
            elif oof.shape[1:] != test.shape[1:]:
                skipped[name] = f"width mismatch {oof.shape} vs {test.shape}"
            elif not (np.isfinite(oof).all() and np.isfinite(test).all()):
                skipped[name] = "non-finite values"
            else:
                names.append(name)
                oofs.append(oof)
                tests.append(test)
    return ExternalPredictions(names, oofs, tests, skipped)


def load_oof_banks(
    dirs: Sequence[str | Path], *, n_train: int, n_test: int
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Recursively collect binary OOF/test pairs in the layouts public banks use.

    Extracted from s6e11's ``pool_stack.load_banks`` (the loader behind every
    pooled stack there). Recognised layouts, searched with ``rglob``:

    - ``<base>_oof.npy`` + ``<base>_full.npy`` (or ``_test``) -> ``<parent>:<base>``
    - ``oof_<base>.npy`` + ``test_<base>.npy`` (or ``refit_``/``foldavg_``)
      -> ``<parent>:<base>``
    - a directory holding ``oof.npy`` + ``test.npy`` -> ``<parent>``

    Two-column (n, 2) arrays are reduced to the positive-class column. Pairs
    with the wrong row counts, non-finite values or unreadable files are
    skipped silently; the first pair found under a name wins. Unlike
    :func:`load_public_predictions` this is binary-only and does not report
    skips. **Row order is not checked:** confirm the bank's author sorted rows
    the way ``build_features()`` does before stacking.
    """
    out: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*.npy")):
            name: str | None = None
            t_path: Path | None = None
            if p.name.endswith("_oof.npy"):
                base = p.name[: -len("_oof.npy")]
                t_path = _first_existing(
                    p.parent, [f"{base}_{s}.npy" for s in ("full", "test")]
                )
                name = f"{p.parent.name}:{base}"
            elif p.name.startswith("oof_"):
                base = p.name[len("oof_") : -len(".npy")]
                t_path = _first_existing(
                    p.parent, [f"{s}_{base}.npy" for s in ("test", "refit", "foldavg")]
                )
                name = f"{p.parent.name}:{base}"
            elif p.name == "oof.npy":
                t_path = p.parent / "test.npy"
                name = p.parent.name
            if name is None or t_path is None or not t_path.exists():
                continue
            try:
                o = np.load(p).astype(np.float64)
                t = np.load(t_path).astype(np.float64)
            except (ValueError, OSError):
                continue
            if o.ndim == 2 and o.shape[1] == 2:  # noqa: PLR2004
                o = o[:, 1]
                t = t[:, 1] if t.ndim == 2 else t  # noqa: PLR2004
            o, t = o.ravel(), t.ravel()
            ok = len(o) == n_train and len(t) == n_test and name not in out
            if ok and np.isfinite(o).all() and np.isfinite(t).all():
                out[name] = (o, t)
    return out


def _first_existing(parent: Path, candidates: Sequence[str]) -> Path | None:
    return next((parent / c for c in candidates if (parent / c).exists()), None)


class NestedStack(NamedTuple):
    """Result of :func:`nested_lr_stack`: AUC of the OOF logits, OOF and test logits."""

    auc: float
    oof: NDArray[np.float64]
    test: NDArray[np.float64] | None


def nested_lr_stack(
    x: NDArray[np.float64],
    y: NDArray[Any],
    x_test: NDArray[np.float64] | None = None,
    *,
    c: float = 0.01,
    n_splits: int = 5,
    seeds: Sequence[int] = (7, 13, 99),
    cols: Sequence[int] | NDArray[np.int64] | None = None,
    max_iter: int = 5000,
) -> NestedStack:
    """Binary L2 logistic-regression stack scored by nested CV.

    For each seed an LR is fit on K-1 folds of ``StratifiedKFold(n_splits)``
    and its decision function scores the held-out fold; OOF logits are
    averaged over seeds, test logits over every (seed, fold) fit. The split
    differs from the members' own CV folds, so the score is honest with
    respect to the meta fit.

    s6e11 lessons behind the defaults: with 100+ rank-logit members
    ``C = 0.01`` beat 0.003-1; the free LR beat non-negative weights, ridge,
    pairwise-ranking, clustered-member, segment-interaction, local and GBDT
    residual stackers; nest 10 equalled nest 5; and the paired-bootstrap sd of
    a pool delta was 0.00004, so differences below ~1e-4 are noise.

    Args:
        x: OOF member matrix (n, m), e.g. :func:`rank_logit` columns.
        y: Binary labels.
        x_test: Optional test matrix with the same columns.
        c: Inverse L2 strength.
        n_splits: Folds of the nested split.
        seeds: One split per seed.
        cols: Column subset to use (default all), handy for ablations.
        max_iter: LR solver iterations.
    """
    use = np.arange(x.shape[1]) if cols is None else np.asarray(cols)
    oof = np.zeros(len(y))
    tst = None if x_test is None else np.zeros(len(x_test))
    n_fits = len(seeds) * n_splits
    for s in seeds:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=s).split(x, y)
        for trn, va in splits:
            m = LogisticRegression(C=c, max_iter=max_iter).fit(x[trn][:, use], y[trn])
            oof[va] += m.decision_function(x[va][:, use]) / len(seeds)
            if tst is not None and x_test is not None:
                tst += m.decision_function(x_test[:, use]) / n_fits
    return NestedStack(float(roc_auc_score(y, oof)), oof, tst)
