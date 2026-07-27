"""Pre-registration #4, cell C: AVERAGE-then-STACK vs STACK-then-AVERAGE.

The deployed #1 averages first and stacks second: each te3-family leg is a mean over 8
seeds, the mean is clipped-logit'd, and the LR meta is fit on that. The alternative order
— fit one stack per seed on that seed's legs, then average the 8 stacks — has never been
measured. The two are NOT equivalent, because `clipped_logit` and the meta are nonlinear:

    meta( mean_s p_s )   !=   mean_s meta( p_s )

This is an AGGREGATE-level structural choice (it is a property of the pipeline, not of any
one member), so Day-21's screening rule #3 does not discount it — which is precisely why
it is worth measuring once the per-member class has been priced out. Day-18's
`probe_pool_geom` swept the pooling FUNCTION for members of a leg (logit / median /
trimmed vs arithmetic); it did not touch the pooling POSITION relative to the combiner.

It is also the one cell where the campaign's two big mechanisms make OPPOSING guesses:
  * veto #18's mechanism ("the LR meta is itself an averaging operator") predicts the
    order barely matters — the meta re-extracts whatever the pre-averaging did;
  * rule #3's mechanism (per-row noise drives argmax deltas) predicts stack-then-average
    could be BETTER, since each per-seed meta sees a noisier, more diverse input and the
    final average then removes meta-level noise as well as leg-level noise.
Either way the answer is informative, which is the bar for spending the CPU.

ARMS (the standard 12 split-half cells: meta + decision weights refit on half A, scored
on half B; the 5 non-breadth legs are held fixed in both arms so the only difference is
the position of the seed average)
  avg_then_stack   deployed: 5 fixed legs + the 3 eight-seed breadth legs -> one meta
  stack_then_avg   for each seed s: 5 fixed legs + the 3 legs AT SEED s -> its own meta;
                   average the 8 metas' probabilities, then the decision layer
  stack_then_avg4  same at 4 seeds, to expose the trajectory in the number of stacks

PRE-REGISTERED BARS (CLAUDE.md "PRE-REGISTRATION #4"), raised by the freeze:
  >= +0.0005  report to the user as a PROPOSAL to reopen the freeze, with costs stated
  <= +0.0001  cell CLOSED BY MEASUREMENT
  otherwise   inconclusive, no action

Run: python src/probe_stack_order.py -> results/probe_stack_order.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777]
FIXED = ["lgbm_bagw", "hgbc_bagw", "xgboost_bagw", "catboost_bagw", "mlp"]
BREADTH = ["hgbc_te3_u_breadth", "lgbm_te3_u_breadth", "realmlp_te13_u_breadth"]
PERSEED = ["hgbc_te3_u_s{s}", "lgbm_te3_u_s{s}", "realmlp_te13_u_s{s}"]
BAR_PROPOSE = 0.0005
BAR_CLOSE = 0.0001


def _fit_meta(
    X: NDArray[np.float32], y: NDArray[np.int64], fit_idx: NDArray[np.int64]
) -> LogisticRegression:
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return lr


def _decide(
    y: NDArray[np.int64],
    fit_idx: NDArray[np.int64],
    eval_idx: NDArray[np.int64],
    p_fit: NDArray[np.float64],
    p_eval: NDArray[np.float64],
) -> float:
    dw = np.asarray(tune_decision_weights(y[fit_idx], p_fit))
    return float(balanced_accuracy_score(y[eval_idx], weighted_predict(p_eval, dw)))


def cell(
    X_avg: NDArray[np.float32],
    X_per: list[NDArray[np.float32]],
    y: NDArray[np.int64],
    fit_idx: NDArray[np.int64],
    eval_idx: NDArray[np.int64],
) -> dict[str, float]:
    out: dict[str, float] = {}
    lr = _fit_meta(X_avg, y, fit_idx)
    out["avg_then_stack"] = _decide(
        y, fit_idx, eval_idx, lr.predict_proba(X_avg[fit_idx]), lr.predict_proba(X_avg[eval_idx])
    )
    pf = np.zeros((len(fit_idx), 3))
    pe = np.zeros((len(eval_idx), 3))
    for i, Xs in enumerate(X_per, start=1):
        lrs = _fit_meta(Xs, y, fit_idx)
        pf += lrs.predict_proba(Xs[fit_idx])
        pe += lrs.predict_proba(Xs[eval_idx])
        if i == 4:
            out["stack_then_avg4"] = _decide(y, fit_idx, eval_idx, pf / i, pe / i)
    out["stack_then_avg"] = _decide(y, fit_idx, eval_idx, pf / len(X_per), pe / len(X_per))
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y

    def blk(key: str) -> NDArray[np.float32]:
        return clipped_logit(np.load(RESULTS_DIR / f"oof_{key}.npy")).astype(np.float32)

    fixed = [blk(k) for k in FIXED]
    X_avg = np.hstack(fixed + [blk(k) for k in BREADTH])
    X_per = [np.hstack(fixed + [blk(t.format(s=s)) for t in PERSEED]) for s in SEEDS]

    cells: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=3)(delayed(cell)(X_avg, X_per, y, f, e) for f, e in cells)
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}

    lines = [
        "=" * 78,
        "probe_stack_order — AVERAGE-then-STACK (deployed) vs STACK-then-AVERAGE",
        "the pooling POSITION relative to the nonlinear combiner; never measured",
        f"{len(cells)} split-half cells; 5 fixed legs + 3 te3-family legs over {len(SEEDS)} seeds",
        f"PRE-REG BARS vs avg_then_stack: >=+{BAR_PROPOSE} propose reopening the freeze / "
        f"<=+{BAR_CLOSE} closed by measurement",
        "=" * 78,
        "",
    ]
    for a in sorted(arms, key=lambda k: -scores[k].mean()):
        d = scores[a] - scores["avg_then_stack"]
        lines.append(
            f"  {a:<16} {scores[a].mean():.6f}   vs deployed {d.mean():+.6f} "
            f"(sd {d.std(ddof=1):.6f}, se {d.std(ddof=1) / np.sqrt(len(d)):.6f}, "
            f"{int((d > 0).sum())}/{len(d)} positive)"
        )

    dm = float((scores["stack_then_avg"] - scores["avg_then_stack"]).mean())
    lines += ["", f"HEADLINE  stack_then_avg - avg_then_stack = {dm:+.6f}"]
    if dm >= BAR_PROPOSE:
        lines.append("VERDICT: PROPOSE reopening the freeze — report to the user, do not act.")
    elif dm <= BAR_CLOSE:
        lines.append(
            "VERDICT: CLOSED BY MEASUREMENT — the pooling POSITION is not a lever either; "
            "veto #18's 'the meta is itself an averaging operator' wins over rule #3's "
            "opposing guess. Finals unchanged."
        )
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no action.")

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_stack_order.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
