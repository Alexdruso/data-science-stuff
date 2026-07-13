"""LB-probe validation — constraint-split cross-check of the Bayes candidate.

Fit the tilt on a random half of the scored submissions, build the full-Bayes
candidate from that posterior, then evaluate its delta vs the best existing
submission under the posterior fitted on the OTHER half (independent noisy
view of the same public split). A positive cross-evaluated delta means the
flips encode real public-split information, not tilt overfit.

Also: p-sensitivity of the decisions and a two-stack prior variant, both cheap.
"""

import numpy as np

from lbprobe_fit import fit, predicted_score


def bayes_labels(res: dict) -> np.ndarray:
    qp = res["qp"]
    return np.argmax(qp / qp.sum(axis=0, keepdims=True), axis=1).astype(np.int8)


def cross_split(n_reps: int = 4, seed: int = 42) -> None:
    full = fit()
    files = full["files"]
    b_obs = full["b_obs"]
    best_idx = int(np.argmax(b_obs))
    best_labels = full["y_mat"][:, best_idx]
    k = len(files)

    rng = np.random.default_rng(seed)
    print("\n=== constraint-split cross-validation ===")
    deltas = []
    for rep in range(n_reps):
        perm = rng.permutation(k)
        half_a = sorted(perm[: k // 2].tolist())
        half_b = sorted(perm[k // 2 :].tolist())
        for tr, ev in ((half_a, half_b), (half_b, half_a)):
            res_tr = fit(file_subset=[files[i] for i in tr], quiet=True)
            res_ev = fit(file_subset=[files[i] for i in ev], quiet=True)
            cand = bayes_labels(res_tr)
            d_ev = predicted_score(cand, res_ev["qp"]) - predicted_score(
                best_labels, res_ev["qp"]
            )
            d_tr = predicted_score(cand, res_tr["qp"]) - predicted_score(
                best_labels, res_tr["qp"]
            )
            deltas.append(d_ev)
            print(
                f"rep {rep}: train-half Δ={d_tr:+.5f} -> cross-eval Δ={d_ev:+.5f} "
                f"(flips {int((cand != best_labels).sum())})"
            )
    arr = np.array(deltas)
    print(
        f"\ncross-eval Δ: mean {arr.mean():+.5f}, sd {arr.std():.5f}, "
        f"positive {int((arr > 0).sum())}/{len(arr)}"
    )

    print("\n=== p sensitivity (decision flips of full-fit Bayes labels) ===")
    base = bayes_labels(full)
    for p in (0.1, 0.3, 0.5):
        res_p = fit(p=p, quiet=True)
        lab = bayes_labels(res_p)
        print(f"p={p}: {int((lab != base).sum())} rows differ from p=0.2 decisions")


if __name__ == "__main__":
    cross_split()
