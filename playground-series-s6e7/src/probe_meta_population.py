"""Pre-registration #4, cell A: the LR meta's fitting POPULATION — the last empty cell
in the combiner coverage matrix.

Day-13's coverage matrix swept the combiner's hypothesis class (gbdt/nn metas),
regularization (C in {0.3, 1, 3}), fitting objective (direct-bacc NM polish), calibration
shape (per-class power) and structure (pairwise coupling, band regimes), and concluded
"the LR stack is a measured local optimum across every combiner axis". It never varied
**which rows the meta is fit on**.

Day-20 attacked exactly that question one stage LATER — for the decision layer — and its
opening line was "every prior probe varied the OPTIMIZER while holding the fitting
POPULATION fixed at the unweighted OOF rows". That observation applies verbatim to the
meta, and nobody noticed. Mechanism: adv-AUC 0.65, so the meta's coefficients are fit on
a train x-population that is not the test one; balanced accuracy's recalls are
expectations over P(x | y=c), so the population matters in principle.

This is an AGGREGATE-level cell, so Day-21's screening rule #3 (per-member gains decay
~1/M) does NOT discount it — which is exactly why it is worth the CPU now that the
per-member class is priced out.

ARMS (same 12 split-half cells as every seat test: meta AND decision weights refit on
half A, scored on half B)
  lr       incumbent — LR C=1.0, balanced class weight, unweighted rows
  lr_adv   identical, but row-weighted by the density ratio p_test(x)/p_train(x) from the
           cached cross-fitted adversarial-validation scores, q99.5-clipped and
           mean-normalised — the SAME recipe Day-20 used for the decision layer, so the
           two population probes are directly comparable
  lr_advd  meta AND decision weights both fitted under the adv weighting (the meta alone
           may be rescued or masked by an unweighted decision layer downstream)

PRE-REGISTERED BARS (CLAUDE.md "PRE-REGISTRATION #4"), raised by the freeze because
adoption would change the deployed, byte-verified, already-submitted #1:
  >= +0.0005  report to the user as a PROPOSAL to reopen the freeze, with costs stated
  <= +0.0001  cell CLOSED BY MEASUREMENT — the combiner coverage matrix is then complete
              on the population axis too
  otherwise   inconclusive, no action

Honest prior: ~0. `probe_covariate_shift` measured importance weighting on the BASE model
as flat because the shift is *pure covariate* (P(y|x) unchanged), and `probe_dw_shift`
measured the adv-weighted decision layer at +0.00004 honest. If P(y|x) really is
shift-invariant, a reweighted meta fit has nothing to find either.

Run: python src/probe_meta_population.py -> results/probe_meta_population.txt
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
BAR_PROPOSE = 0.0005
BAR_CLOSE = 0.0001
KEYS8 = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "mlp",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
    "realmlp_te13_u_breadth",
]


def adv_row_weights(adv: NDArray[np.float64], n_train: int, n_test: int) -> NDArray[np.float64]:
    """Day-20's recipe verbatim: density ratio, size-corrected, q99.5-clipped, mean 1."""
    odds = adv / np.clip(1.0 - adv, 1e-9, None)
    ratio = odds * (n_train / n_test)
    row_w = np.clip(ratio, 0.0, np.quantile(ratio, 0.995))
    return row_w / row_w.mean()


def cell(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    row_w: NDArray[np.float64],
    fit_idx: NDArray[np.int64],
    eval_idx: NDArray[np.int64],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, use_w, dw_w in (("lr", False, False), ("lr_adv", True, False), ("lr_advd", True, True)):
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        lr.fit(
            X[fit_idx], y[fit_idx], sample_weight=row_w[fit_idx] if use_w else None
        )
        pf, pe = lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])
        if dw_w:
            # weighted decision layer: resample-free approximation — tune on the fit half
            # with rows replicated in expectation via the same density ratio
            idx = fit_idx
            ww = row_w[idx]
            keep = np.random.default_rng(0).random(len(idx)) < (ww / ww.max())
            dw = np.asarray(tune_decision_weights(y[idx][keep], pf[keep]))
        else:
            dw = np.asarray(tune_decision_weights(y[fit_idx], pf))
        # ALWAYS scored unweighted on the held-out half: the deliverable is plain BA
        out[name] = float(balanced_accuracy_score(y[eval_idx], weighted_predict(pe, dw)))
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    X = np.hstack(
        [clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32) for k in KEYS8]
    )
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    row_w = adv_row_weights(adv, len(y), len(ds.test))
    ess = float(row_w.sum() ** 2 / (row_w**2).sum())

    cells: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=3)(delayed(cell)(X, y, row_w, f, e) for f, e in cells)
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}

    lines = [
        "=" * 78,
        "probe_meta_population — the LR meta's FITTING POPULATION",
        "(the one cell the Day-13 combiner coverage matrix never swept)",
        f"{len(cells)} split-half cells; legs = {KEYS8}",
        f"adv row weights: sd {row_w.std():.3f}  ESS {ess:,.0f}/{len(y):,} "
        f"({100 * ess / len(y):.1f}%)",
        f"PRE-REG BARS vs lr: >=+{BAR_PROPOSE} propose reopening the freeze / "
        f"<=+{BAR_CLOSE} closed by measurement",
        "always SCORED unweighted on the held-out half — the deliverable is plain BA",
        "=" * 78,
        "",
    ]
    for a in sorted(arms, key=lambda k: -scores[k].mean()):
        d = scores[a] - scores["lr"]
        lines.append(
            f"  {a:<8} {scores[a].mean():.6f}   vs lr {d.mean():+.6f} "
            f"(sd {d.std(ddof=1):.6f}, se {d.std(ddof=1) / np.sqrt(len(d)):.6f}, "
            f"{int((d > 0).sum())}/{len(d)} positive)"
        )

    best = max((a for a in arms if a != "lr"), key=lambda k: (scores[k] - scores["lr"]).mean())
    dm = float((scores[best] - scores["lr"]).mean())
    lines += ["", f"HEADLINE  best challenger = {best}, delta {dm:+.6f}"]
    if dm >= BAR_PROPOSE:
        lines.append("VERDICT: PROPOSE reopening the freeze — report to the user, do not act.")
    elif dm <= BAR_CLOSE:
        lines.append(
            "VERDICT: CLOSED BY MEASUREMENT — the meta's fitting population is not a lever; "
            "the combiner coverage matrix is now complete on the population axis. "
            "Finals unchanged."
        )
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no action.")

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_meta_population.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
