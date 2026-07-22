"""Test-transform TE-bagging probe pair (Day-18 #1) — the one lever recorded
UNREADABLE offline (Day-16, probe_te_design): it changes only test-side
predictions, so no OOF instrument can see it. This script converts it into a
single quota submission whose public delta vs an already-scored reference is
EXACT (balanced accuracy is linear in per-row assignments — lbprobe finding).

Lever, operationalized (pre-registered): each fold model's test rows are
encoded with that fold's full-map TE (fit on its 80% split). Bagging = replace
that per-fold test encoding with the MEAN of all 5 folds' test encodings
(~sqrt(5) variance reduction of the map noise; train side untouched, hence
OOF-invisible by construction). No new encoder family, no new smoothing — the
fold axis is the only map-diversity source, so this is the canonical bag.

Reference: hgbc_te3_u_s42.csv — reliable public score on file, solo
TE-consumer, regeneration validated by the Day-16 referee run (~5 min CPU).

Composition = DELTA TRANSFER (sklearn 1.9's TargetEncoder internal CV is
unseeded-shuffled, so a retrain cannot reproduce the deployed model
byte-for-byte): retrain the 5 fold models, measure the lever's per-row proba
delta on them (bag minus stock, same fitted model both arms), and apply that
delta to the DEPLOYED test_hgbc_te3_u_s42.npy. The stock arm then reproduces
the deployed CSV exactly by construction, and the probe differs from the
scored reference ONLY by the lever's estimated effect (retrain noise enters
Δleg at second order, reported, not confounded into the composition).

Read (pre-registered, LB-blind protocol): user submits the probe CSV and
reports Δpub = pub(probe) − pub(reference on file). Adoption rule: bagging
enters the TE-consumer finals lineage only if Δpub > 0 AND Δpub ≥ 2×sd_null
(flip-count MC below); anything else = lever measured-closed. Known weak
link, stated up front: within-lineage near-clone Δpub→Δpriv transfer is ~0
(Day-9), so a pass additionally needs the flip fingerprint to look like
variance reduction (missing-driver-concentrated), not boundary re-drawing.

Run: python src/probe_tebag_compose.py
  → submissions/probe_tebag_hgbc_te3_u_s42.csv
  → results/probe_tebag_preread.txt
Resumable per fold via results/cache/ckpt_tebag/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    CLASSES,
    N_CLASSES,
    N_FOLDS,
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    load_dataset,
)
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle_utils import weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
LEG = "hgbc_te3_u_s42"
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
P_PUBLIC = 0.2  # lbprobe_common.P_PUBLIC_DEFAULT
MC_DRAWS = 200_000
CKPT = CACHE_DIR / "ckpt_tebag"


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None and N_FOLDS == 5, "u surface, K=5 only"
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    # ---- exact trainer preprocessing (train_hgbc_te3, u surface) ----
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
            axis=1,
        )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(ds.y, ds.y))

    # ---- pass 1: per-fold full-map test encodings (deterministic given split) ----
    CKPT.mkdir(parents=True, exist_ok=True)
    enc_cache = CKPT / "test_encodings.npz"
    if enc_cache.exists():
        T = np.load(enc_cache)["T"]
        say("pass 1: loaded cached per-fold test encodings")
    else:
        T = np.zeros((N_FOLDS, len(ds.test_ids), len(te_names)), dtype=np.float32)
        for f, (tr_idx, _val_idx) in enumerate(folds, 1):
            enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
            enc.fit(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
            T[f - 1] = enc.transform(te_te_view).astype(np.float32)
            print(f"  pass 1 fold {f}: test encoding done", flush=True)
        np.savez_compressed(enc_cache, T=T)
    T_bag = T.mean(axis=0)
    enc_shift = float(np.abs(T - T_bag[None]).mean())
    say(f"per-fold vs bagged test-encoding mean |Δ|: {enc_shift:.6f}")

    # ---- pass 2: retrain fold models, predict both arms with the SAME model ----
    n_test = len(ds.test_ids)
    stock_sum = np.zeros((n_test, N_CLASSES))
    bag_sum = np.zeros((n_test, N_CLASSES))
    oof_retrain = np.zeros((len(ds.y), N_CLASSES))
    for f, (tr_idx, val_idx) in enumerate(folds, 1):
        fck = CKPT / f"f{f}.npz"
        if fck.exists():
            z = np.load(fck)
            p_stock, p_bag, p_val = z["stock"], z["bag"], z["val"]
            say(f"  fold {f}: loaded checkpoint")
        else:
            enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
            te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
            model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=SEED)
            model.fit(assemble(X.iloc[tr_idx], te_tr), ds.y[tr_idx])
            assert list(model.classes_) == list(range(N_CLASSES))
            p_val = model.predict_proba(
                assemble(X.iloc[val_idx], enc.transform(te_tr_view.iloc[val_idx]))
            )
            p_stock = model.predict_proba(assemble(X_test, T[f - 1]))
            p_bag = model.predict_proba(assemble(X_test, T_bag))
            np.savez_compressed(fck, stock=p_stock, bag=p_bag, val=p_val)
            say(f"  fold {f}: n_iter={model.n_iter_}")
            del model
        stock_sum += p_stock / N_FOLDS
        bag_sum += p_bag / N_FOLDS
        oof_retrain[val_idx] = p_val

    # ---- retrain fidelity (context for attribution, not a hard gate) ----
    from sklearn.metrics import balanced_accuracy_score

    with (RESULTS_DIR / f"decision_weights_{LEG}.json").open() as fh:
        w_map = json.load(fh)
    weights = np.array([w_map[c] for c in CLASSES])
    oof_dep = np.load(RESULTS_DIR / f"oof_{LEG}.npy")
    say(
        f"retrain OOF argmax {balanced_accuracy_score(ds.y, oof_retrain.argmax(1)):.5f} "
        f"/ wt {balanced_accuracy_score(ds.y, weighted_predict(oof_retrain, weights)):.5f}  "
        f"(deployed twin argmax {balanced_accuracy_score(ds.y, oof_dep.argmax(1)):.5f} "
        f"/ wt {balanced_accuracy_score(ds.y, weighted_predict(oof_dep, weights)):.5f})"
    )
    test_dep = np.load(RESULTS_DIR / f"test_{LEG}.npy")
    lab_dep = weighted_predict(test_dep, weights)
    m_noise = int((weighted_predict(stock_sum, weights) != lab_dep).sum())
    say(f"retrain-noise label mismatch (retrained stock vs deployed): {m_noise} rows")

    # ---- composition gate: stock arm must reproduce the deployed CSV exactly ----
    ref = pd.read_csv(SUBMISSIONS_DIR / f"{LEG}.csv")
    assert (ref["id"].to_numpy() == ds.test_ids).all(), "reference CSV row order"
    ref_lab = ref[TARGET].to_numpy()
    gate = int((np.array(CLASSES)[lab_dep] != ref_lab).sum())
    say(f"composition gate (deployed npy + weights vs reference CSV): {gate} mismatches")
    assert gate == 0, "composition path does not reproduce the scored reference"

    # ---- delta transfer: apply the lever's estimated effect to deployed probas ----
    d_leg = bag_sum - stock_sum
    say(f"Δleg |proba| mean {np.abs(d_leg).mean():.6f} / max {np.abs(d_leg).max():.4f}")
    probe_lab = weighted_predict(test_dep + d_leg, weights)
    flips = probe_lab != lab_dep
    n_flips = int(flips.sum())
    say(f"\nlever flips (probe vs reference): {n_flips} rows "
        f"({100 * n_flips / n_test:.3f}% of test)")
    trans = pd.crosstab(
        pd.Series(np.array(CLASSES)[lab_dep[flips]], name="from"),
        pd.Series(np.array(CLASSES)[probe_lab[flips]], name="to"),
    )
    say(trans.to_string())
    md_test = ds.test[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    say(f"missing-driver share of flips: {md_test[flips].mean():.3f} "
        f"(base rate {md_test.mean():.3f})")

    # ---- null MC: Δpub distribution if flips carry no signal ----
    rng = np.random.default_rng(SEED)
    prior = np.bincount(ds.y, minlength=N_CLASSES) / len(ds.y)
    n_pub_c = prior * n_test * P_PUBLIC  # E[public class counts], train-prior approx
    fr, to = lab_dep[flips], probe_lab[flips]
    in_pub = rng.random((MC_DRAWS, n_flips)) < P_PUBLIC
    truth_is_to = rng.random((MC_DRAWS, n_flips)) < 0.5  # null: truth ∈ {from,to} 50/50
    gain = np.where(truth_is_to, 1.0 / (3.0 * n_pub_c[to]), -1.0 / (3.0 * n_pub_c[fr]))
    delta = (in_pub * gain).sum(axis=1)
    sd_null = float(delta.std())
    say(f"\nnull MC (p_pub={P_PUBLIC}, {MC_DRAWS} draws): "
        f"E[Δpub]={delta.mean():+.6f}  sd={sd_null:.6f}")
    say(f"ADOPTION RULE (pre-registered): Δpub > 0 AND Δpub ≥ 2×sd_null = "
        f"{2 * sd_null:+.6f}; else lever CLOSED.")

    out = SUBMISSIONS_DIR / "probe_tebag_hgbc_te3_u_s42.csv"
    pd.DataFrame({"id": ds.test_ids, TARGET: np.array(CLASSES)[probe_lab]}).to_csv(
        out, index=False
    )
    say(f"\nprobe submission → {out}")
    say(f"read = pub(probe) − pub({LEG}.csv, score on file); user submits + reports.")
    (RESULTS_DIR / "probe_tebag_preread.txt").write_text("\n".join(lines) + "\n")
    print(f"report → {RESULTS_DIR / 'probe_tebag_preread.txt'}", flush=True)


if __name__ == "__main__":
    main()
