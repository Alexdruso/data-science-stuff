"""Gate the missingness-mechanism repairs on the re-masked instrument (plan #0b).

Same fold loop and val surfaces as diag_mask_shift.py — identical fold splits
(seed 42) and identical val remask RNG streams (1000/2000 + fold), so scores are
exactly paired with the baseline instrument. The difference: TRAINING folds are
repaired before fitting.

  r2a -- in-place uniform re-mask: Bernoulli(r_col) masking of observed values in
         the 4 shifted columns at the TEST marginal rates. Training rows then
         carry uniform NaNs on top of the trigger NaNs -- the same mixture the
         testmech val surface has, so the training mechanism matches the
         (emulated) test mechanism and the NaN<->trigger coupling is broken.
  r2b -- concat augmentation: original training fold + a re-masked copy (2x rows,
         2x fit time; keeps every clean row intact alongside its masked twin).

Baseline (no repair) reference = results/diag_mask_shift.txt:
  plain 0.9487 / testmech 0.9467 / control 0.9477 (weighted).
GATE: repaired testmech weighted bacc >= baseline testmech + 0.001, with the
plain surface not cratered.

Also reported: decision weights fit on the model's own TESTMECH OOF and applied
to the testmech surface -- the "fix the weight-fitting surface" half-repair,
deployable on its own even if retraining is a wash.

Run: ../.venv/bin/python src/diag_repair.py r2a | tee results/diag_repair_r2a.txt
Env: S6E7_REPAIR_MULT (float, default 1.0) scales the re-mask rates.

Day-7 addition — the TESTVOL read (volume-honest, Day-6 adv-AUC 0.6886 follow-up):
train/test marginal NaN rates are ~EQUAL per column, so both the r2a repair and the
testmech val surface roughly double NaN volume in the 4 columns vs true test. But for
val rows COMPLETE in all 4 columns, the testmech re-mask is exactly the true-test
surface: uniform mechanism at exact test volume. "testvol" = testmech restricted to
that subset (~86% of rows; biased toward the non-triggered population, identical
subset across mult runs -> paired). Compare mult runs on it; no stored baseline.

Day-20 addition — mode `te13` (E1.2, the never-run comparison at the CURRENT
recipe): the r2a/r2a_te chains measured the repair on the Day-4 plain-LGBM and
Day-8 numeric-TE recipes only. te13 (all-13-columns string TargetEncoder, the
deployed Susol HGBC config) makes NaN its OWN TE level, so the shifted columns'
'nan' posteriors are estimated from the trigger population -- a mechanism that
could make the shift WORSE, not neutral. This mode holds the val inputs fixed at
the test mechanism and varies ONLY the training surface:

  S6E7_REPAIR_MULT=0.0 -> no-repair arm (rates 0 => remask is a no-op)
  S6E7_REPAIR_MULT=1.0 -> deployed R2a repair
  S6E7_REPAIR_MULT=0.5 -> the Day-7 half-rate optimum

All arms share fold splits (seed 42) and the val-remask RNG streams (1000/2000 +
fold), so the TESTVOL column is exactly paired across them. Encoder + model are
fit on the (repaired) fold-train rows; the three val surfaces are mapped with the
SAME full-fold encoder. No test predictions (read-only diagnostic); per-fold val
probas are checkpointed (this box hard-reboots).
PRE-REGISTERED BAR: repaired minus no-repair >= +0.0006 on TESTVOL, with the
plain-surface delta >= -0.0002.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from diag_mask_shift import TRIGGERS, remask, test_rates
from features import NUM_COLS
from train_te_num import encode_fold, with_te
from train_common import (
    N_CLASSES,
    RESULTS_DIR,
    Dataset,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle_utils import weighted_predict

# weighted-bacc reference surfaces from results/diag_mask_shift.txt (no repair)
BASELINE = {"plain": 0.9487, "testmech": 0.9467, "control": 0.9477}
GATE = 0.001


def repair_train(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    rates: dict[str, float],
    rng: np.random.Generator,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the repaired training fold for `mode` ('r2a'|'r2a_te'|'te13'|'r2b')."""
    remasked = remask(X_tr, rates, rng, "testmech")
    if mode in ("r2a", "r2a_te", "te13"):
        return remasked, y_tr
    if mode == "r2b":
        X_aug = pd.concat([X_tr, remasked], ignore_index=True)
        return X_aug, np.concatenate([y_tr, y_tr])
    raise SystemExit(f"unknown repair mode {mode!r} (expected r2a|r2a_te|te13|r2b)")


# --------------------------------------------------------------------------
# mode te13: the deployed Susol recipe (train_hgbc_te3.py) as the consumer
# --------------------------------------------------------------------------


def _te13_prep(
    df: pd.DataFrame, cat_cols: list[str], cats: dict[str, list[str]]
) -> pd.DataFrame:
    """train_hgbc_te3's base matrix: cat NaN -> 'missing' Categorical, nums raw."""
    out = df.copy()
    for col in cat_cols:
        s = out[col].astype(object).fillna("missing").astype(str)
        out[col] = pd.Categorical(s, categories=cats[col])
    return out


def te13_vocab(ds: Dataset) -> dict[str, list[str]]:
    """Fixed categorical vocabulary over train+test (+ the 'missing' level).

    The repair/remask only turns values INTO NaN, so this vocabulary covers every
    surface (plain, repaired, testmech, control) — no surface-dependent category
    order can leak into the model.
    """
    out: dict[str, list[str]] = {}
    for col in ds.cat_cols:
        vals = set(ds.train[col].dropna().unique()) | set(ds.test[col].dropna().unique())
        out[col] = sorted({str(v) for v in vals} | {"missing"})
    return out


def te13_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    val_frames: dict[str, pd.DataFrame],
    raw_cols: list[str],
    cat_cols: list[str],
    cats: dict[str, list[str]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit TE+HGBC on the (repaired) fold-train rows; map every val surface.

    The encoder is cross-fitted on the training rows (fit_transform's inner CV,
    invariant #3) and the val surfaces are transformed with the SAME full-fold
    maps — so the only thing varying across arms is the training surface.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import TargetEncoder

    from train_hgbc_te3 import HGBC_CONFIG

    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    base_tr = _te13_prep(X_tr, cat_cols, cats)
    view_tr = base_tr[raw_cols].astype(str)
    enc = TargetEncoder(
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        smooth="auto",
        target_type="multiclass",
    )
    blk_tr = enc.fit_transform(view_tr, y_tr)

    def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
        )

    model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
    model.fit(assemble(base_tr, blk_tr), y_tr)
    assert list(model.classes_) == list(range(N_CLASSES)), model.classes_

    out: dict[str, np.ndarray] = {}
    for name, frame in val_frames.items():
        base_v = _te13_prep(frame, cat_cols, cats)
        blk_v = enc.transform(base_v[raw_cols].astype(str))
        out[name] = model.predict_proba(assemble(base_v, blk_v))
    return out


def _surface_masks(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Recompute (changed, complete4) — deterministic given the fixed RNG streams."""
    n = len(ds.y)
    changed = np.zeros(n, dtype=bool)
    complete4 = np.zeros(n, dtype=bool)
    cols = [c for c in TRIGGERS if c in ds.train.columns]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_tr, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        X_val = ds.train.iloc[val_idx]
        vt = remask(X_val, test_rates(), np.random.default_rng(1000 + fold), "testmech")
        changed[val_idx] = (
            vt[cols].isna().to_numpy() & X_val[cols].notna().to_numpy()
        ).any(axis=1)
        complete4[val_idx] = X_val[cols].notna().all(axis=1).to_numpy()
    return changed, complete4


def te13_report() -> None:
    """E1.2 paired comparison across the te13 arms (m000/m050/m100).

    Reads the per-arm OOF arrays saved by `diag_repair.py te13` and each arm's
    deployment-style decision weights (parsed from its run log), then scores every
    arm on the identical row subsets. TESTVOL (val rows complete in the 4 shifted
    columns, remasked at exact test volume) is the decisive column.
    """
    import re

    ds: Dataset = load_dataset()
    changed, c4 = _surface_masks(ds)
    arms = [("m000", "no repair"), ("m050", "repair mult 0.5"), ("m100", "repair mult 1.0")]
    bar_tv, bar_plain = 0.0006, -0.0002

    print("E1.2 — repair x te13: val inputs FIXED at test mechanism, training")
    print("surface varied. Consumer = HGBC (deployed Susol config) + 13-col string")
    print("TargetEncoder, 5 folds (seed 42), shared fold splits and val-remask RNG.")
    print(f"\nTESTVOL rows (complete-in-4): {c4.sum():,} ({c4.mean():.1%})   "
          f"newly-NaN'd rows: {changed.sum():,} ({changed.mean():.1%})")

    rows: dict[str, dict[str, float]] = {}
    probs: dict[str, dict[str, np.ndarray]] = {}
    for arm, _label in arms:
        tag = "" if arm == "m100" else f"_{arm}"
        try:
            oof = {
                m: np.load(RESULTS_DIR / f"oof_repair_te13{tag}_{m}.npy")
                for m in ("plain", "testmech", "control")
            }
        except FileNotFoundError as exc:
            print(f"\n[{arm}] MISSING: {exc}")
            continue
        log = (RESULTS_DIR / f"log_diag_te13_{arm}.txt").read_text()
        mt = re.search(r"weights \(fit on plain OOF\): \[([^\]]+)\]", log)
        assert mt, f"no decision weights in log_diag_te13_{arm}.txt"
        w = np.array([float(v) for v in mt.group(1).split()])
        probs[arm] = oof
        rows[arm] = {
            "plain": balanced_accuracy_score(ds.y, weighted_predict(oof["plain"], w)),
            "testmech": balanced_accuracy_score(
                ds.y, weighted_predict(oof["testmech"], w)
            ),
            "control": balanced_accuracy_score(
                ds.y, weighted_predict(oof["control"], w)
            ),
            "testvol": balanced_accuracy_score(
                ds.y[c4], weighted_predict(oof["testmech"][c4], w)
            ),
            "testvol_argmax": balanced_accuracy_score(
                ds.y[c4], oof["testmech"][c4].argmax(1)
            ),
            "plainvol": balanced_accuracy_score(
                ds.y[c4], weighted_predict(oof["plain"][c4], w)
            ),
            "changed": balanced_accuracy_score(
                ds.y[changed], weighted_predict(oof["testmech"][changed], w)
            ),
            "w0": w[0], "w1": w[1], "w2": w[2],
        }

    hdr = ["arm", "plain", "testmech", "control", "TESTVOL", "tv-argmax", "changed"]
    print("\n" + " ".join(f"{h:>10}" for h in hdr))
    for arm, label in arms:
        if arm not in rows:
            continue
        r = rows[arm]
        print(f"{arm:>10} " + " ".join(
            f"{r[k]:>10.5f}" for k in
            ["plain", "testmech", "control", "testvol", "testvol_argmax", "changed"]
        ) + f"   ({label}; weights "
            f"{r['w0']:.3f}/{r['w1']:.3f}/{r['w2']:.3f})")

    # Day-4 damage decomposition, recomputed at the te13 recipe:
    #   plain - testmech = total emulated damage
    #   control - testmech = PLACEMENT (uniform vs on-trigger at equal NaN count)
    #   plain - control   = VOLUME (information actually deleted)
    print("\ndamage decomposition (Day-4 diag_mask_shift semantics, per arm):")
    print(f"{'arm':>10} {'total(pl-tm)':>13} {'placement(ct-tm)':>18} "
          f"{'volume(pl-ct)':>14}")
    for arm, _label in arms:
        if arm not in rows:
            continue
        r = rows[arm]
        print(f"{arm:>10} {r['plain'] - r['testmech']:>+13.5f} "
              f"{r['control'] - r['testmech']:>+18.5f} "
              f"{r['plain'] - r['control']:>+14.5f}")
    print("Day-4 reference (plain LGBM, no TE): total -0.0020 = placement -0.0010")
    print("                                     + volume -0.0010")

    if "m000" not in rows:
        print("\nno m000 arm -> no delta table")
        return
    base = rows["m000"]
    print(f"\ndeltas vs m000 (no repair) — bar: TESTVOL >= {bar_tv:+.4f} "
          f"AND plain >= {bar_plain:+.4f}")
    print(f"{'arm':>10} {'d TESTVOL':>11} {'d tv-argmax':>12} {'d plain':>10} "
          f"{'d testmech':>11} {'d changed':>11} {'verdict':>10}")
    for arm, _label in arms:
        if arm == "m000" or arm not in rows:
            continue
        r = rows[arm]
        d_tv = r["testvol"] - base["testvol"]
        d_pl = r["plain"] - base["plain"]
        ok = "PASS" if (d_tv >= bar_tv and d_pl >= bar_plain) else "FAIL"
        print(f"{arm:>10} {d_tv:>+11.5f} "
              f"{r['testvol_argmax'] - base['testvol_argmax']:>+12.5f} "
              f"{d_pl:>+10.5f} {r['testmech'] - base['testmech']:>+11.5f} "
              f"{r['changed'] - base['changed']:>+11.5f} {ok:>10}")

    # paired row-bootstrap of the TESTVOL delta (same rows resampled for both arms)
    rng = np.random.default_rng(20260724)
    y4 = ds.y[c4]
    print("\npaired row-bootstrap of the TESTVOL delta (200 reps, plain-fit weights):")
    for arm, _label in arms:
        if arm == "m000" or arm not in rows:
            continue
        wa = np.array([rows[arm]["w0"], rows[arm]["w1"], rows[arm]["w2"]])
        wb = np.array([base["w0"], base["w1"], base["w2"]])
        pa = weighted_predict(probs[arm]["testmech"][c4], wa)
        pb = weighted_predict(probs["m000"]["testmech"][c4], wb)
        deltas = []
        for _ in range(200):
            idx = rng.integers(0, len(y4), len(y4))
            deltas.append(
                balanced_accuracy_score(y4[idx], pa[idx])
                - balanced_accuracy_score(y4[idx], pb[idx])
            )
        d = np.array(deltas)
        print(f"  {arm}: mean {d.mean():+.5f}  sd {d.std():.5f}  "
              f"P(delta>0) {(d > 0).mean():.2f}  "
              f"95% CI [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "r2a"
    if mode == "te13_report":
        te13_report()
        return
    mult = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))

    ds: Dataset = load_dataset()
    rates = {c: r * mult for c, r in test_rates().items()}
    print(f"repair mode: {mode}  rate mult: {mult}")
    print(f"re-mask rates: { {k: round(v, 4) for k, v in rates.items()} }")

    X = ds.train.copy()
    if mode != "te13":  # te13 keeps raw object cats (its own 'missing' prep)
        for col in ds.cat_cols:
            X[col] = X[col].astype("category")
    params = load_params()
    cats = te13_vocab(ds) if mode == "te13" else {}
    ck_dir = RESULTS_DIR / "cache" / f"ckpt_diagrep_{mode}_m{int(round(mult * 100)):03d}"
    if mode == "te13":
        ck_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds.y)
    oof = {m: np.zeros((n, N_CLASSES)) for m in ["plain", "testmech", "control"]}
    changed = np.zeros(n, dtype=bool)  # rows that received >=1 new NaN (testmech)
    complete4 = np.zeros(n, dtype=bool)  # rows fully observed in the 4 shifted cols

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        X_tr, y_tr = repair_train(
            X.iloc[tr_idx], ds.y[tr_idx], rates, np.random.default_rng(3000 + fold), mode
        )
        if fold == 1:  # overshoot diagnostic: repaired-train NaN volume vs test
            print("  repaired-train NaN rates vs test marginal:")
            for col in TRIGGERS:
                if col in X.columns:
                    print(f"    {col:<26} {X_tr[col].isna().mean():.4f} "
                          f"vs test {test_rates()[col]:.4f}")
        # identical val surfaces to diag_mask_shift.py (same RNG streams)
        X_val = X.iloc[val_idx]
        val_test = remask(X_val, test_rates(), np.random.default_rng(1000 + fold), "testmech")
        val_ctrl = remask(X_val, test_rates(), np.random.default_rng(2000 + fold), "control")

        cols = [c for c in TRIGGERS if c in X.columns]
        changed[val_idx] = (
            val_test[cols].isna().to_numpy() & X_val[cols].notna().to_numpy()
        ).any(axis=1)
        complete4[val_idx] = X_val[cols].notna().all(axis=1).to_numpy()

        if mode == "te13":
            surfaces = {"plain": X_val, "testmech": val_test, "control": val_ctrl}
            ck = {m: ck_dir / f"f{fold}_{m}.npy" for m in surfaces}
            if all(p.exists() for p in ck.values()):
                probas = {m: np.load(p) for m, p in ck.items()}
                print(f"  fold {fold}: loaded checkpoint")
            else:
                probas = te13_fold(
                    X_tr, y_tr, surfaces, ds.feature_cols, ds.cat_cols, cats, 42
                )
                for m, p in ck.items():
                    np.save(p, probas[m])
            for m, p in probas.items():
                oof[m][val_idx] = p
            b = balanced_accuracy_score(ds.y[val_idx], probas["plain"].argmax(1))
            print(f"  fold {fold}: plain argmax bacc {b:.4f}", flush=True)
            continue

        X_val_fit = X_val
        if mode == "r2a_te":
            # Exact-value numeric TE (train_te_num protocol: inner cross-fit for the
            # training rows, full-fold maps for val/test) computed on the REPAIRED
            # matrix — exactly what a mult-scaled TE rebuild would deploy. Val
            # surfaces are TE-mapped with the same full-fold maps so the paired
            # TESTVOL/testmech reads stay comparable to the m050/m100 chains.
            num_cols = [c for c in NUM_COLS if c in X.columns]
            # encode_fold only reads numeric columns; the repair remasks bmi (and
            # water, which TE excludes), so hand it a numeric-only frame whose
            # training rows carry the repaired values.
            X_rep = X[num_cols].copy()
            X_rep.iloc[tr_idx] = X_tr[num_cols].to_numpy()
            te_tr, full_maps, prior = encode_fold(X_rep, ds.y, tr_idx, num_cols, 42)
            X_tr = pd.concat([X_tr, te_tr], axis=1)
            X_val_fit = with_te(X_val, full_maps, prior)
            val_test = with_te(val_test, full_maps, prior)
            val_ctrl = with_te(val_ctrl, full_maps, prior)
            X_val = X_val_fit

        model = LGBMClassifier(**{**params, "random_state": 42})
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val_fit, ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES))

        for m, xv in [("plain", X_val), ("testmech", val_test), ("control", val_ctrl)]:
            oof[m][val_idx] = model.predict_proba(xv)
        b = balanced_accuracy_score(ds.y[val_idx], oof["plain"][val_idx].argmax(1))
        print(f"  fold {fold}: plain argmax bacc {b:.4f} (best_iter {model.best_iteration_})")

    tag = "" if mult == 1.0 else f"_m{int(round(mult * 100)):03d}"
    for m, arr in oof.items():
        np.save(RESULTS_DIR / f"oof_repair_{mode}{tag}_{m}.npy", arr)

    # deployment-style: weights fit on the repaired model's PLAIN oof
    weights = robust_decision_weights(ds.y, oof["plain"])
    print(f"\ndecision weights (fit on plain OOF): {weights.round(4)}")
    if mode == "te13":
        print("(the 'vs-baseline' column below references the Day-4 plain-LGBM run —")
        print(" meaningless for te13; the decisive read is arm-vs-arm on TESTVOL.)")
    print(f"\n{'surface':<10} {'argmax':>8} {'weighted':>9} {'vs-baseline':>12}")
    scores = {}
    for m in ["plain", "testmech", "control"]:
        a = balanced_accuracy_score(ds.y, oof[m].argmax(1))
        w = balanced_accuracy_score(ds.y, weighted_predict(oof[m], weights))
        scores[m] = w
        print(f"{m:<10} {a:>8.4f} {w:>9.4f} {w - BASELINE[m]:>+12.4f}")

    # half-repair: fit the weights on the testmech surface instead
    weights_tm = robust_decision_weights(ds.y, oof["testmech"])
    tm_tm = balanced_accuracy_score(ds.y, weighted_predict(oof["testmech"], weights_tm))
    print(f"\nweights fit on TESTMECH oof -> testmech surface: {tm_tm:.4f} "
          f"(vs plain-fit {scores['testmech']:.4f}; weights {weights_tm.round(4)})")

    # TESTVOL: volume-honest read = testmech restricted to rows complete in the
    # 4 shifted columns (uniform mechanism at EXACT test volume there). Paired
    # across mult runs (same subset, same RNG streams); no stored baseline.
    c4 = complete4
    print(f"\nTESTVOL (complete-in-4 rows: {c4.sum():,} = {c4.mean():.1%}):")
    for wname, wvec in [("plain-fit", weights), ("testmech-fit", weights_tm)]:
        tv = balanced_accuracy_score(
            ds.y[c4], weighted_predict(oof["testmech"][c4], wvec)
        )
        pv = balanced_accuracy_score(
            ds.y[c4], weighted_predict(oof["plain"][c4], wvec)
        )
        print(f"  {wname:<13} testvol={tv:.4f}  (plain-surface same rows: {pv:.4f})")
        print(f"  {'':<13} testvol6={tv:.6f}   plain6={pv:.6f}")

    if mode != "te13":
        delta = max(scores["testmech"], tm_tm) - BASELINE["testmech"]
        verdict = "PASS" if delta >= GATE else "FAIL"
        print(f"\nGATE (testmech vs baseline {BASELINE['testmech']:.4f}, "
              f"need >=+{GATE}): {delta:+.4f} -> {verdict}")
        print(f"plain-surface cost vs baseline: "
              f"{scores['plain'] - BASELINE['plain']:+.4f}")
    else:
        print("\nGATE: arm-vs-arm (see results/diag_repair_te13.txt); this run's")
        print(f"  headline numbers: TESTVOL(plain-fit) / plain / testmech = "
              f"{balanced_accuracy_score(ds.y[c4], weighted_predict(oof['testmech'][c4], weights)):.6f}"
              f" / {scores['plain']:.6f} / {scores['testmech']:.6f}")

    print(f"\nrows with >=1 new NaN (testmech): {changed.sum():,} ({changed.mean():.3%})")
    for m in ["plain", "testmech"]:
        pred = weighted_predict(oof[m][changed], weights)
        acc = float((pred == ds.y[changed]).mean())
        bacc = balanced_accuracy_score(ds.y[changed], pred)
        print(f"  {m:<10} on-changed-rows acc={acc:.4f} bacc={bacc:.6f}")
    # secondary read, volume-honest: changed rows that were complete in the 4
    # (the rows where the remask IS the true test surface AND something moved)
    cc = changed & complete4
    pred = weighted_predict(oof["testmech"][cc], weights)
    print(f"  changed & complete-in-4: {cc.sum():,} rows  "
          f"testmech bacc={balanced_accuracy_score(ds.y[cc], pred):.6f}")


if __name__ == "__main__":
    main()
