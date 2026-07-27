"""Day-20 — is the DECISION LAYER tuned on the wrong population?

New lens (surface-mismatch at the decision layer, not new signal). Every prior
decision-layer probe varied the OPTIMIZER (n_bags 8->32, direct-bacc NM polish,
Bayes cost certificate, per-class power calib, regime/band metas, joint
assignment) while holding the POPULATION the weights are fitted on fixed: the
unweighted OOF rows. This probe attacks the population instead.

Mechanism
---------
Balanced accuracy is the mean of class-conditional recalls,
``BA = (1/3) sum_c E_{x ~ P(x|y=c)}[ 1(rule(x) = c) ]``.

1. **Label shift is a NON-lever, analytically.** Recalls are conditional on the
   class, so a pure change of the class prior pi(y) leaves every recall (and
   therefore BA, and therefore the BA-optimal weight vector) unchanged. Any
   Saerens-EM / BBSE prior correction is a no-op for THIS metric. Part 1 states
   the estimate anyway (with a null band) because a large prior move would be
   evidence about the joint shift, and because the claim deserves a number.
2. **Covariate/joint shift IS a lever.** Recalls are expectations over
   ``P(x | y=c)``, and the campaign has measured a real train->test shift
   (adv-AUC 0.65; gender share 32.5%->29.2%; numeric shifts; Day-16/17 referee
   confirmed the generator ADDED cross-column dependence). If ``P(x|y=c)``
   differs, the w that maximizes train-population recalls is not the w that
   maximizes test-population recalls. Part 2 retunes w against the test-like
   population via importance weights from the existing cross-fitted adversarial
   scores, and measures the gain HONESTLY (weights fitted on one half of the
   OOF rows, weighted BA read on the held-out half, 5 seeded splits).
3. Part 3 is a free diagnostic for the sibling hypothesis (posterior SHARPNESS
   mismatch: each OOF row is scored by ONE fold model, each test row by an
   average of 5), with the covariate-shift confound controlled by restricting
   train rows to the test-like tail.

Pre-registered bars
-------------------
* Part 1 closes (as predicted) unless ``|pi_hat_test - pi_train|`` exceeds the
  bootstrap null band by z > 3 on a minority class -- and even then it is only
  evidence about the joint, not an adoptable lever.
* Part 2 escalates only if the held-out adv-weighted BA gain of adv-tuned w over
  unweighted-tuned w is **>= +0.0010 with a consistent sign across all 5
  splits** (the campaign's informative-read rule: >= 2x read noise, and >= the
  0.001 deployment gate). Anything smaller is recorded and closed.
* Part 3 is diagnostic only: no adoption, it either motivates a nested-CV
  follow-up or kills it.

Read-only: loads cached OOF/test arrays + the deployed weights. Writes one
report, no submissions, no retraining.

Run: python src/probe_dw_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import tune_decision_weights

KEYS = ["lrstack_u4_r", "hgbc_te3_u_breadth"]
N_SPLITS = 5
N_BOOT_NULL = 200
REPORT = RESULTS_DIR / "probe_dw_shift.txt"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def weighted_bacc(
    y: NDArray[np.int64], pred: NDArray[np.int64], row_w: NDArray[np.float64]
) -> float:
    """Mean of importance-weighted per-class recalls (plain BA when row_w == 1)."""
    recalls = []
    for c in range(len(CLASSES)):
        m = y == c
        denom = row_w[m].sum()
        recalls.append(float((row_w[m] * (pred[m] == c)).sum() / denom) if denom else 0.0)
    return float(np.mean(recalls))


def apply_w(proba: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.asarray(np.argmax(proba * w, axis=1), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Part 1 -- label-shift estimate (expected to be a non-lever; see docstring)
# --------------------------------------------------------------------------- #
def em_prior(
    proba: NDArray[np.float64], pi_train: NDArray[np.float64], n_iter: int = 200
) -> NDArray[np.float64]:
    """Saerens-Latinne-Decaestecker EM estimate of the target class prior."""
    pi = pi_train.copy()
    for _ in range(n_iter):
        adj = proba * (pi / pi_train)
        adj /= adj.sum(axis=1, keepdims=True)
        new = adj.mean(axis=0)
        if np.max(np.abs(new - pi)) < 1e-10:
            pi = new
            break
        pi = new
    return np.asarray(pi, dtype=np.float64)


def bbse_prior(
    oof: NDArray[np.float64],
    y: NDArray[np.int64],
    test: NDArray[np.float64],
    w: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Black-box shift estimation: pi_test = C^-1 q, C[k,c] = P(yhat=k | y=c)."""
    n_c = len(CLASSES)
    pred_oof, pred_test = apply_w(oof, w), apply_w(test, w)
    conf = np.zeros((n_c, n_c))
    for c in range(n_c):
        m = y == c
        for k in range(n_c):
            conf[k, c] = float((pred_oof[m] == k).mean())
    q = np.array([float((pred_test == k).mean()) for k in range(n_c)])
    pi = np.linalg.solve(conf, q)
    return np.asarray(pi, dtype=np.float64), float(np.linalg.cond(conf))


def part1(
    key: str,
    oof: NDArray[np.float64],
    test: NDArray[np.float64],
    y: NDArray[np.int64],
    w: NDArray[np.float64],
    rng: np.random.Generator,
) -> list[str]:
    pi_train = np.array([(y == c).mean() for c in range(len(CLASSES))])
    pi_em = em_prior(test, pi_train)
    pi_bbse, cond = bbse_prior(oof, y, test, w)

    # Null band: EM on train-prior samples of test size, using the OOF surface.
    boots = np.array(
        [
            em_prior(oof[rng.integers(0, len(y), len(test))], pi_train)
            for _ in range(N_BOOT_NULL)
        ]
    )
    # EM on the OOF surface at the train prior is itself biased (posteriors are
    # not perfectly calibrated), so the null band is centred on the bootstrap
    # mean, not on pi_train.
    null_mu, null_sd = boots.mean(axis=0), boots.std(axis=0)
    z = (pi_em - null_mu) / np.maximum(null_sd, 1e-12)

    lines = [f"--- Part 1 label shift [{key}] ---"]
    lines.append(f"pi_train        = {np.round(pi_train, 5).tolist()}  ({CLASSES})")
    lines.append(f"mean OOF post.  = {np.round(oof.mean(axis=0), 5).tolist()}")
    lines.append(f"mean test post. = {np.round(test.mean(axis=0), 5).tolist()}")
    lines.append(f"EM pi_test      = {np.round(pi_em, 5).tolist()}")
    lines.append(f"EM null mu/sd   = {np.round(null_mu, 5).tolist()} / {np.round(null_sd, 5).tolist()}")
    lines.append(f"EM z vs null    = {np.round(z, 2).tolist()}")
    lines.append(f"BBSE pi_test    = {np.round(pi_bbse, 5).tolist()} (cond(C)={cond:.1f})")
    # BBSE null band: resample OOF rows to test size (train prior by construction)
    # and re-run the estimator, so the shift gets a sampling error bar.
    n_c = len(CLASSES)
    pred_oof = apply_w(oof, w)
    conf = np.zeros((n_c, n_c))
    for c in range(n_c):
        m = y == c
        for k in range(n_c):
            conf[k, c] = float((pred_oof[m] == k).mean())
    b_boot = []
    for _ in range(N_BOOT_NULL):
        idx = rng.integers(0, len(y), len(test))
        q = np.array([float((pred_oof[idx] == k).mean()) for k in range(n_c)])
        b_boot.append(np.linalg.solve(conf, q))
    b_boot = np.asarray(b_boot)
    b_mu, b_sd = b_boot.mean(axis=0), b_boot.std(axis=0)
    lines.append(
        f"BBSE null mu/sd = {np.round(b_mu, 5).tolist()} / {np.round(b_sd, 5).tolist()}"
        f"  => z {np.round((pi_bbse - b_mu) / np.maximum(b_sd, 1e-12), 2).tolist()}"
    )
    lines.append(
        f"BBSE shift vs pi_train (abs pp) = "
        f"{np.round(100 * (pi_bbse - pi_train), 3).tolist()}"
    )
    lines.append(
        "NOTE: BA = mean of class-conditional recalls => prior-invariant; a real "
        "prior move is evidence about the joint shift, NOT an adoptable lever."
    )
    return lines


# --------------------------------------------------------------------------- #
# Part 2 -- adv-weighted (test-like population) decision-weight tuning
# --------------------------------------------------------------------------- #
def _one_split(
    oof: NDArray[np.float64],
    y: NDArray[np.int64],
    row_w: NDArray[np.float64],
    tr: NDArray[np.int64],
    va: NDArray[np.int64],
    w_dep: NDArray[np.float64],
    seed: int,
) -> tuple[float, float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Fit w on `tr` (adv-weighted and unweighted); read adv-weighted BA on `va`."""
    y_tr, p_tr, rw_tr = y[tr], oof[tr], row_w[tr]
    y_va, p_va, rw_va = y[va], oof[va], row_w[va]

    w_adv = tune_decision_weights(
        y_tr, p_tr, seed=seed, score_fn=lambda yt, yp: weighted_bacc(yt, yp, rw_tr)
    )
    w_plain = tune_decision_weights(y_tr, p_tr, seed=seed)

    ba_adv = weighted_bacc(y_va, apply_w(p_va, w_adv), rw_va)
    ba_plain = weighted_bacc(y_va, apply_w(p_va, w_plain), rw_va)
    ba_dep = weighted_bacc(y_va, apply_w(p_va, w_dep), rw_va)
    return ba_adv, ba_plain, ba_dep, w_adv, w_plain


def part2(
    key: str,
    oof: NDArray[np.float64],
    test: NDArray[np.float64],
    y: NDArray[np.int64],
    w_dep: NDArray[np.float64],
    adv: NDArray[np.float64],
    n_test: int,
) -> list[str]:
    # density ratio p_test(x)/p_train(x) from the cross-fitted AV classifier,
    # corrected for the train/test size ratio in its training set, clipped for ESS.
    odds = adv / np.clip(1.0 - adv, 1e-9, None)
    ratio = odds * (len(y) / n_test)
    hi = np.quantile(ratio, 0.995)
    row_w = np.clip(ratio, 0.0, hi)
    row_w = row_w / row_w.mean()
    ess = float(row_w.sum() ** 2 / (row_w**2).sum())

    lines = [f"--- Part 2 adv-weighted decision weights [{key}] ---"]
    lines.append(
        f"row_w: mean 1.000 sd {row_w.std():.3f} q99.5-clip {hi:.2f} "
        f"ESS {ess:,.0f}/{len(y):,} ({100 * ess / len(y):.1f}%)"
    )

    skf = StratifiedKFold(2, shuffle=True, random_state=0)
    jobs = []
    for s in range(N_SPLITS):
        skf = StratifiedKFold(2, shuffle=True, random_state=100 + s)
        tr, va = next(iter(skf.split(oof, y)))
        jobs.append((tr, va, 42 + s))
    out = Parallel(n_jobs=min(N_SPLITS, 5))(
        delayed(_one_split)(oof, y, row_w, tr, va, w_dep, sd) for tr, va, sd in jobs
    )

    d_vs_plain, d_vs_dep = [], []
    for i, (ba_adv, ba_plain, ba_dep, w_adv, w_plain) in enumerate(out):
        d_vs_plain.append(ba_adv - ba_plain)
        d_vs_dep.append(ba_adv - ba_dep)
        lines.append(
            f"  split {i}: heldout adv-wBA  adv-tuned {ba_adv:.6f} | "
            f"plain-tuned {ba_plain:.6f} | deployed {ba_dep:.6f}  "
            f"(d_plain {ba_adv - ba_plain:+.6f}, d_dep {ba_adv - ba_dep:+.6f})  "
            f"w_adv {np.round(w_adv, 4).tolist()} w_plain {np.round(w_plain, 4).tolist()}"
        )
    dp, dd = np.array(d_vs_plain), np.array(d_vs_dep)
    lines.append(
        f"HONEST mean d(adv-tuned - plain-tuned) = {dp.mean():+.6f} "
        f"(sd {dp.std(ddof=1):.6f}, signs {[int(np.sign(v)) for v in dp]})"
    )
    lines.append(
        f"HONEST mean d(adv-tuned - deployed)    = {dd.mean():+.6f} "
        f"(sd {dd.std(ddof=1):.6f}, signs {[int(np.sign(v)) for v in dd]})"
    )
    verdict = (
        "ESCALATE" if dp.mean() >= 0.0010 and np.all(dp > 0) else "SUB-BAR -> close"
    )
    lines.append(f"PRE-REGISTERED BAR +0.0010 & consistent sign => {verdict}")

    # full-data adv-tuned weights: how different, how many test rows would move?
    w_full = tune_decision_weights(
        y, oof, seed=42, score_fn=lambda yt, yp: weighted_bacc(yt, yp, row_w)
    )
    flips = int((apply_w(test, w_full) != apply_w(test, w_dep)).sum())
    lines.append(
        f"full-data adv-tuned w = {np.round(w_full, 4).tolist()} vs deployed "
        f"{np.round(w_dep, 4).tolist()}; test-row flips {flips:,}/{len(test):,} "
        f"({100 * flips / len(test):.2f}%)"
    )
    lines.append(
        f"unweighted OOF BA: deployed {weighted_bacc(y, apply_w(oof, w_dep), np.ones(len(y))):.6f}"
        f" | adv-tuned {weighted_bacc(y, apply_w(oof, w_full), np.ones(len(y))):.6f}"
    )
    return lines


# --------------------------------------------------------------------------- #
# Part 3 -- posterior sharpness: OOF (1 fold model) vs test (5-model average)
# --------------------------------------------------------------------------- #
def part3(
    key: str,
    oof: NDArray[np.float64],
    test: NDArray[np.float64],
    adv: NDArray[np.float64],
) -> list[str]:
    def sharp(p: NDArray[np.float64]) -> tuple[float, float]:
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
        return float(p.max(axis=1).mean()), float(ent.mean())

    # control the covariate-shift confound: compare test against the most
    # test-like train rows (top decile of the AV score), not all train rows.
    tl = adv >= np.quantile(adv, 0.90)
    mo, eo = sharp(oof)
    mt, et = sharp(test)
    ml, el = sharp(oof[tl])
    return [
        f"--- Part 3 sharpness diagnostic [{key}] ---",
        f"OOF  (1 fold model/row): mean max-p {mo:.5f}  mean entropy {eo:.5f}",
        f"OOF test-like decile   : mean max-p {ml:.5f}  mean entropy {el:.5f}",
        f"TEST (5-model average) : mean max-p {mt:.5f}  mean entropy {et:.5f}",
        f"gap vs test-like decile: max-p {mt - ml:+.5f}  entropy {et - el:+.5f}",
        "(gap ~0 => averaging does not sharpen the surface => the 'weights tuned on "
        "a noisier surface' hypothesis has no mechanism and needs no nested CV)",
    ]


def main() -> None:
    ds = load_dataset()
    y = ds.y
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    assert adv.shape == y.shape, (adv.shape, y.shape)
    rng = np.random.default_rng(0)

    lines = [
        "Day-20 probe: decision layer vs population/surface mismatch",
        f"adv scores: cross-fitted train-vs-test AV (plain surface), n={len(adv):,}, "
        f"mean {adv.mean():.4f}",
        "",
    ]
    for key in KEYS:
        oof = np.load(RESULTS_DIR / f"oof_{key}.npy")
        test = np.load(RESULTS_DIR / f"test_{key}.npy")
        dwj = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
        # JSON is {class_name: weight}; CLASSES order is the array column order.
        w_dep = np.asarray([dwj[c] for c in CLASSES], dtype=np.float64)
        lines += part1(key, oof, test, y, w_dep, rng) + [""]
        lines += part3(key, oof, test, adv) + [""]
        lines += part2(key, oof, test, y, w_dep, adv, len(test)) + [""]

    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
