"""PP-1 E[max] finals portfolio + PP-2 paired-delta calculus (Day-9, prob purist).

The two-finals game scores max(S1, S2) on the PRIVATE split. For near-equal
means, E[max] ~ mu + 0.4*sigma_Delta — so zero-mean variance ADDED to final #2
(decorrelating it from #1) raises the expected max. The 220:1 lever: flipping an
indifference-band row from at-risk to a minority class moves that class's
recall by 1/n_minority_private vs 1/n_atrisk_private — variance scales with the
square (~220x for fit). At the tuned decision boundary the expected score
change of a flip is ~0 (first-order condition of the weight tuning), so the
perturbation is ~zero-mean with capped downside.

Instrument: cell-based Monte Carlo. Population = OOF rows with TRUE labels;
per rep, a 295,753-row test emulation is drawn (multinomial over cells =
(true class x correct_1 x correct_2)), then the private 80% is drawn by
vectorized multivariate hypergeometric. Captures label uncertainty + split
randomness AND the S1/S2 correlation exactly. B=20k reps per K.

PP-2: same machinery on (lrstack6_r, metablend_r) and (lrstack6_r,
final3_realmlp): joint (Delta_pub, Delta_priv) distribution — how much does a
public tiebreak between near-clones pin the private ordering?

Report -> results/probe_emax_portfolio.txt. Candidate flipped CSVs are written
ONLY as candidates (user decides finals).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES, TARGET
from train_common import RESULTS_DIR, SUBMISSIONS_DIR, load_dataset

N_TEST = 295_753
PUB_FRAC = 0.2
N_PUB = round(N_TEST * PUB_FRAC)  # 59,151
N_PRIV = N_TEST - N_PUB  # 236,602
B = 20_000
K_GRID = [500, 1_000, 2_000, 4_000, 6_000, 8_000, 12_000]
RNG = np.random.default_rng(20260711)


def load_weights(name: str) -> NDArray[np.float64]:
    with (RESULTS_DIR / f"decision_weights_{name}.json").open() as f:
        w = json.load(f)
    if "decision" in w:  # nested blend+decision format (e.g. final3_realmlp)
        w = w["decision"]
    return np.array([w[c] for c in CLASSES])


def champion_oof_labels(y: NDArray[np.int64]) -> NDArray[np.int64]:
    """v1b recipe on the _v1 arrays (forensics_v1b_v2 protocol), OOF side."""
    cache = RESULTS_DIR / "champion_oof_labels.npy"
    if cache.exists():
        return np.load(cache)
    from scipy.optimize import minimize
    from sklearn.metrics import balanced_accuracy_score

    from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

    gbdts = ["lgbm", "xgboost", "catboost"]
    oof = {m: np.load(RESULTS_DIR / f"oof_{m}_v1.npy") for m in gbdts}
    w = {m: np.asarray(tune_decision_weights(y, oof[m])) for m in gbdts}
    pc = [oof[m] * w[m] / (oof[m] * w[m]).sum(axis=1, keepdims=True) for m in gbdts]

    def blend(wv: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.exp(wv) / np.exp(wv).sum()
        return sum(s[i] * a for i, a in enumerate(pc))

    res = minimize(
        lambda wv: -float(balanced_accuracy_score(y, blend(wv).argmax(axis=1))),
        np.ones(len(gbdts)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    blended = blend(res.x)
    dw = tune_decision_weights(y, blended)
    lab = np.asarray(weighted_predict(blended, dw), dtype=np.int64)
    print(f"champion v1b OOF reconstruction: {balanced_accuracy_score(y, lab):.4f} "
          "(scored config was 0.9494)")
    np.save(cache, lab)
    return lab


def corrected(name: str, split: str) -> NDArray[np.float64]:
    proba = np.load(RESULTS_DIR / f"{split}_{name}.npy")
    return proba * load_weights(name)


def labels_of(scores: NDArray[np.float64]) -> NDArray[np.int64]:
    return scores.argmax(axis=1).astype(np.int64)


def flip_labels(
    scores: NDArray[np.float64], k: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Flip the k at-risk-predicted rows nearest the decision boundary to their
    runner-up minority class. Returns (labels, flipped_row_idx)."""
    lab = labels_of(scores)
    ar = np.where(lab == 0)[0]
    runner = 1 + scores[ar, 1:].argmax(axis=1)
    margin = scores[ar, 0] - scores[ar, 1:].max(axis=1)
    order = np.argsort(margin)[:k]
    out = lab.copy()
    out[ar[order]] = runner[order]
    return out, ar[order]


def mc_cells(
    y: NDArray[np.int64], lab_a: NDArray[np.int64], lab_b: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Joint MC of (bacc_a, bacc_b) on emulated (public, private) splits.

    Cells = (true class, corr_a, corr_b): 12 counts. Per rep: test counts ~
    multinomial(N_TEST, oof cell freqs); private counts ~ mv-hypergeometric.
    Returns (bacc_a_priv, bacc_b_priv, bacc_a_pub, bacc_b_pub), each shape (B,).
    """
    cell = (y * 4 + (lab_a == y) * 2 + (lab_b == y)).astype(np.int64)
    freq = np.bincount(cell, minlength=12).astype(np.float64) / len(y)
    test_counts = RNG.multinomial(N_TEST, freq, size=B)  # (B, 12)

    # sequential vectorized multivariate hypergeometric: draw private counts
    priv = np.zeros((B, 12), dtype=np.int64)
    remaining_total = np.full(B, N_TEST, dtype=np.int64)
    remaining_draw = np.full(B, N_PRIV, dtype=np.int64)
    for c in range(12):
        n_good = test_counts[:, c]
        n_bad = remaining_total - n_good
        draw = RNG.hypergeometric(n_good, np.maximum(n_bad, 0), remaining_draw)
        priv[:, c] = draw
        remaining_total -= n_good
        remaining_draw -= draw
    pub = test_counts - priv

    def bacc(counts: NDArray[np.int64], which: str) -> NDArray[np.float64]:
        acc = np.zeros(B)
        for cls in range(3):
            n_cls = counts[:, cls * 4 : cls * 4 + 4].sum(axis=1)
            if which == "a":
                corr = counts[:, cls * 4 + 2] + counts[:, cls * 4 + 3]
            else:
                corr = counts[:, cls * 4 + 1] + counts[:, cls * 4 + 3]
            acc += corr / np.maximum(n_cls, 1)
        return acc / 3.0

    return bacc(priv, "a"), bacc(priv, "b"), bacc(pub, "a"), bacc(pub, "b")


def part1_emax(y: NDArray[np.int64], out: list[str]) -> int:
    out.append("=" * 72)
    out.append("PP-1  E[max(S1, S2)] with K indifference-band flips in final #2")
    out.append(f"      S1=champion_v1 (v1b reconstruction), S2=lrstack6_r; B={B}, "
               f"private={N_PRIV:,} of {N_TEST:,}")
    out.append("=" * 72)
    lab1 = champion_oof_labels(y)
    s2 = corrected("lrstack6_r", "oof")

    a_p, b_p, _, _ = mc_cells(y, lab1, labels_of(s2))
    base_max = np.maximum(a_p, b_p).mean()
    out.append(
        f"\nbaseline: E[S1]={a_p.mean():.5f} E[S2]={b_p.mean():.5f} "
        f"sd(S1-S2)={np.std(a_p - b_p):.5f}  E[max]={base_max:.5f}"
    )
    out.append(f"P(S2>S1 on private) = {(b_p > a_p).mean():.3f}")

    base_sd_delta = float(np.std(a_p - b_p))
    best_k, best_gain = 0, 0.0
    out.append(f"\n{'K':>6} {'mean dS2':>10} {'sd(S1-S2)':>10} {'E[max]':>9} "
               f"{'gain':>9} {'P(worse)':>9} {'q05 dmax':>9}")
    for k in K_GRID:
        lab2k, _ = flip_labels(s2, k)
        a2, b2, _, _ = mc_cells(y, lab1, lab2k)
        emax = np.maximum(a2, b2).mean()
        d_s2 = b2.mean() - b_p.mean()
        gain = emax - base_max
        # downside = P(bacc(S2K) < baseline E[S2] - 2e-4) on private
        p_worse = float(np.mean(b2 < b_p.mean() - 2e-4))
        q05 = float(np.quantile(np.maximum(a2, b2), 0.05) - np.quantile(np.maximum(a_p, b_p), 0.05))
        out.append(
            f"{k:>6} {d_s2:>+10.5f} {np.std(a2 - b2):>10.5f} {emax:>9.5f} "
            f"{gain:>+9.5f} {p_worse:>9.3f} {q05:>+9.5f}"
        )
        if gain > best_gain:
            best_k, best_gain = k, gain
    out.append(f"(baseline sd(S1-S2) = {base_sd_delta:.5f})")
    out.append(
        f"\nVERDICT PP-1: best K={best_k} with E[max] gain {best_gain:+.5f}"
        if best_gain > 1e-4
        else "\nVERDICT PP-1: no K raises E[max] by >= 0.0001 — flipping not worth it."
    )
    return best_k if best_gain > 1e-4 else 0


def part2_paired_delta(y: NDArray[np.int64], out: list[str]) -> None:
    out.append("\n" + "=" * 72)
    out.append("PP-2  paired-delta calculus: does a public tiebreak pin private order?")
    out.append("=" * 72)
    for a, bname in [("lrstack6_r", "metablend_r"), ("lrstack6_r", "final3_realmlp")]:
        lab_a = labels_of(corrected(a, "oof"))
        lab_b = labels_of(corrected(bname, "oof"))
        n_dis = int((lab_a != lab_b).sum())
        a_p, b_p, a_q, b_q = mc_cells(y, lab_a, lab_b)
        d_priv, d_pub = a_p - b_p, a_q - b_q
        rho = float(np.corrcoef(d_pub, d_priv)[0, 1])
        out.append(
            f"\n{a} vs {bname}: OOF disagreement {n_dis:,} rows "
            f"({n_dis / len(y):.2%})"
        )
        out.append(
            f"  E[d_priv]={d_priv.mean():+.5f} sd={d_priv.std():.5f}   "
            f"E[d_pub]={d_pub.mean():+.5f} sd={d_pub.std():.5f}   corr={rho:+.3f}"
        )
        out.append(f"  unconditional P({a} wins private) = {(d_priv > 0).mean():.3f}")
        for thr in (0.0, 0.0002, 0.0005):
            sel = d_pub > thr
            if sel.sum() > 50:
                out.append(
                    f"  P(wins private | public delta > {thr:.4f}) = "
                    f"{(d_priv[sel] > 0).mean():.3f}   (P(public delta > thr)={sel.mean():.3f})"
                )


def part3_write_candidates(best_k: int, out: list[str]) -> None:
    out.append("\n" + "=" * 72)
    out.append("candidate flipped submissions (test side)")
    out.append("=" * 72)
    import pandas as pd

    ds = load_dataset()
    s2_test = corrected("lrstack6_r", "test")
    # verify reconstruction matches the on-disk submission exactly (row-order guard)
    sub = pd.read_csv(SUBMISSIONS_DIR / "lrstack6_r.csv")
    assert (sub["id"].to_numpy() == ds.test_ids).all(), "id order mismatch"
    rec = np.array(CLASSES)[labels_of(s2_test)]
    agree = (rec == sub[TARGET].to_numpy()).mean()
    out.append(f"reconstruction agreement with lrstack6_r.csv: {agree:.6f}")
    assert agree == 1.0, "test-side reconstruction does not match the submission"

    ks = sorted({k for k in (best_k, best_k // 2) if k > 0})
    for k in ks:
        lab_k, idx = flip_labels(s2_test, k)
        name = f"lrstack6_r_flip{k}.csv"
        pd.DataFrame(
            {"id": ds.test_ids, TARGET: np.array(CLASSES)[lab_k]}
        ).to_csv(SUBMISSIONS_DIR / name, index=False)
        to_fit = int((lab_k[idx] == 1).sum())
        out.append(
            f"  {name}: {k} flips ({to_fit} ->fit, {k - to_fit} ->unhealthy)"
        )


def main() -> None:
    ds = load_dataset()
    out: list[str] = []
    best_k = part1_emax(ds.y, out)
    part2_paired_delta(ds.y, out)
    if best_k:
        part3_write_candidates(best_k, out)
    report = "\n".join(out)
    print(report)
    (RESULTS_DIR / "probe_emax_portfolio.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
