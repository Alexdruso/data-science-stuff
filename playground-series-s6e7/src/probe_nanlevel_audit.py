"""E1.1 — NaN-level composition audit for the te13 recipe (Day-20).

The inherited claim: train masks on 4 columns are deterministic single-column
trigger rules (activity<=>smoking=occasional, water<=>female, bmi<=>sleep=good,
diet<=>gender=other) while TEST masks are uniform.  te13 casts all 13 columns to
strings and target-encodes them, so NaN becomes its OWN TE level:
`TE(physical_activity_level = <nan token>)` is estimated in train from exactly the
occasional-smokers, while in test that level's population is a uniform sample.
This probe measures (zero training):

  (a) train vs test NaN rate BY TRIGGER LEVEL for each shifted column;
  (b) test rows sitting in (col=NaN, trigger=off) cells with ZERO train support;
  (c) the te13 'nan'-level TE vector per column on the plain train matrix vs the
      counterfactual from fitting the SAME encoder on a uniformly-remasked train
      copy (train_common._uniform_remask, the deployed repair plumbing).

PRE-REGISTERED BAR (both must hold to motivate the E1.2 training run):
  >= 1% of test rows in zero-train-support mask combinations
  AND max-over-columns L1 shift of the 'nan'-level TE vector >= 0.02.

Run: ../.venv/bin/python src/probe_nanlevel_audit.py \
        | tee results/probe_nanlevel_audit.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from diag_mask_shift import TRIGGERS
from features import CLASSES
from train_common import (
    MECHANISM_SHIFTED,
    Dataset,
    _uniform_remask,
    load_dataset,
)

BAR_ZERO_SUPPORT = 0.01  # >=1% of test rows
BAR_TE_L1 = 0.02  # max-over-columns L1 shift of the nan-level TE triple


def te_view(df: pd.DataFrame, cat_cols: list[str], cols: list[str]) -> pd.DataFrame:
    """train_hgbc_te3's TE view: cat NaN -> 'missing', then everything -> str."""
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].fillna("missing").astype(str)
    return out[cols].astype(str)


def nan_token(raw: pd.Series, view: pd.Series) -> str:
    """The string token the column's NaNs map to in the TE view."""
    na = raw.isna().to_numpy()
    assert na.any(), "column has no NaN"
    toks = pd.unique(view.to_numpy()[na])
    assert len(toks) == 1, f"ambiguous nan token {toks}"
    return str(toks[0])


def encoding_for(
    enc: TargetEncoder, cols: list[str], col: str, token: str
) -> np.ndarray:
    """The fitted (full-data) encoding triple for one column's category token.

    sklearn's multiclass TargetEncoder stores encodings_ FEATURE-major with the
    class inner: index = feature_j * n_classes + class_c, matching
    get_feature_names_out() == [f"{col}_{k}" for col in cols for k in classes]
    (verified empirically; same order train_hgbc_te3.py assumes for te_names).
    """
    j = cols.index(col)
    cats = list(enc.categories_[j])
    assert token in cats, f"{token!r} not a fitted category of {col}"
    k = cats.index(token)
    n_cls = len(CLASSES)
    return np.array(
        [enc.encodings_[j * n_cls + c][k] for c in range(n_cls)], dtype=float
    )


def fit_encoder(view: pd.DataFrame, y: np.ndarray) -> TargetEncoder:
    """fit() only — the full-data encodings transform() uses (no inner CV needed)."""
    enc = TargetEncoder(smooth="auto", target_type="multiclass")
    enc.fit(view, y)
    return enc


def main() -> None:
    ds: Dataset = load_dataset()
    cols = list(ds.train.columns)
    cat_cols = list(ds.cat_cols)
    y = ds.y
    n_tr, n_te = len(ds.train), len(ds.test)
    print(f"train {n_tr:,} rows x {len(cols)} cols | test {n_te:,} rows")
    print(f"classes: {CLASSES}\n")

    shifted = [c for c in MECHANISM_SHIFTED if c in cols]

    # ---------------------------------------------------------------- (a)
    print("=" * 78)
    print("(a) NaN rate BY TRIGGER LEVEL — train (rule) vs test (uniform?)")
    print("=" * 78)
    for col in shifted:
        trig_col, trig_val = TRIGGERS[col]
        levels = sorted(
            set(ds.train[trig_col].dropna().unique())
            | set(ds.test[trig_col].dropna().unique())
        )
        print(f"\n{col}  (train trigger: {trig_col} == {trig_val!r})")
        print(f"  overall NaN rate: train {ds.train[col].isna().mean():.4f}  "
              f"test {ds.test[col].isna().mean():.4f}")
        print(f"  {'trigger level':<14} {'train n':>10} {'train NaN%':>11} "
              f"{'test n':>10} {'test NaN%':>10}")
        for lv in levels + ["<NA>"]:
            if lv == "<NA>":
                m_tr = ds.train[trig_col].isna().to_numpy()
                m_te = ds.test[trig_col].isna().to_numpy()
            else:
                m_tr = (ds.train[trig_col] == lv).to_numpy()
                m_te = (ds.test[trig_col] == lv).to_numpy()
            if m_tr.sum() == 0 and m_te.sum() == 0:
                continue
            r_tr = ds.train[col].isna().to_numpy()[m_tr].mean() if m_tr.sum() else np.nan
            r_te = ds.test[col].isna().to_numpy()[m_te].mean() if m_te.sum() else np.nan
            mark = "  <- trigger" if lv == trig_val else ""
            print(f"  {str(lv):<14} {m_tr.sum():>10,} {r_tr:>11.4f} "
                  f"{m_te.sum():>10,} {r_te:>10.4f}{mark}")

    # ---------------------------------------------------------------- (b)
    print("\n" + "=" * 78)
    print("(b) TEST rows in (col=NaN, trigger=off) cells — train support")
    print("=" * 78)
    print("cell = (col is NaN) AND (trigger OBSERVED and != trigger value).")
    print("Rows whose TRIGGER is itself masked are counted separately: they are the")
    print("only off-trigger NaN support train actually has.\n")
    print(f"{'column':<26} {'test rows':>11} {'% test':>8} {'train rows':>11} "
          f"{'zero-support':>13} {'train trig-NaN':>15}")
    off_any = np.zeros(n_te, dtype=bool)
    zero_any = np.zeros(n_te, dtype=bool)
    for col in shifted:
        trig_col, trig_val = TRIGGERS[col]
        t_obs_te = ds.test[trig_col].notna().to_numpy()
        t_obs_tr = ds.train[trig_col].notna().to_numpy()
        te_off = (
            ds.test[col].isna().to_numpy()
            & t_obs_te
            & (ds.test[trig_col].to_numpy() != trig_val)
        )
        tr_off = (
            ds.train[col].isna().to_numpy()
            & t_obs_tr
            & (ds.train[trig_col].to_numpy() != trig_val)
        )
        tr_trignan = (ds.train[col].isna().to_numpy() & ~t_obs_tr).sum()
        off_any |= te_off
        zs = int(tr_off.sum()) == 0
        if zs:
            zero_any |= te_off
        print(f"{col:<26} {te_off.sum():>11,} {te_off.mean():>7.2%} "
              f"{tr_off.sum():>11,} {'YES' if zs else 'no':>13} {tr_trignan:>15,}")
    print(f"\nUNION over columns (>=1 off-trigger NaN):   {off_any.sum():,} "
          f"({off_any.mean():.2%} of test)")
    print(f"UNION restricted to ZERO-train-support cells: {zero_any.sum():,} "
          f"({zero_any.mean():.2%} of test)")

    # joint mask pattern check (Day-4 reported 1.25% train-impossible patterns)
    def pattern(df: pd.DataFrame) -> pd.Series:
        return (
            df[shifted].isna().astype(int).astype(str).agg("".join, axis=1)
        )

    p_tr, p_te = pattern(ds.train), pattern(ds.test)
    seen = set(p_tr.unique())
    unseen = ~p_te.isin(seen).to_numpy()
    print(f"joint 4-col mask PATTERNS absent from train:  {unseen.sum():,} "
          f"({unseen.mean():.2%} of test)")

    zero_share = float(zero_any.mean())

    # ---------------------------------------------------------------- (c)
    print("\n" + "=" * 78)
    print("(c) te13 'nan'-level TE vector: plain train vs uniformly-remasked train")
    print("=" * 78)
    view_plain = te_view(ds.train, cat_cols, cols)
    remasked = _uniform_remask(ds.train, ds.test)
    view_rem = te_view(remasked, cat_cols, cols)
    print("remasked-train NaN rates (mult 1.0) vs test marginal:")
    for col in shifted:
        print(f"  {col:<26} {remasked[col].isna().mean():.4f} "
              f"vs test {ds.test[col].isna().mean():.4f}")

    print("\nfitting TargetEncoder (13 string cols, smooth='auto', multiclass) ...")
    enc_plain = fit_encoder(view_plain, y)
    enc_rem = fit_encoder(view_rem, y)
    # sanity: the class-major ordering assumption
    fnames = list(enc_plain.get_feature_names_out())
    assert len(fnames) == len(cols) * len(CLASSES), fnames[:5]
    assert fnames[: len(CLASSES)] == [f"{cols[0]}_{k}" for k in range(len(CLASSES))], (
        f"unexpected encodings_ layout: {fnames[:6]}"
    )
    print(f"  feature_names_out[:4] = {fnames[:4]}  (n={len(fnames)})")

    prior = np.array([(y == k).mean() for k in range(len(CLASSES))])
    print(f"\ntrain class prior: {dict(zip(CLASSES, prior.round(4)))}")
    print(f"\n{'column':<26} {'source':<10} " + " ".join(f"{c:>10}" for c in CLASSES)
          + f" {'L1 vs plain':>12}")
    max_l1 = 0.0
    l1_by_col: dict[str, float] = {}
    for col in shifted:
        tok_p = nan_token(ds.train[col], view_plain[col])
        tok_r = nan_token(remasked[col], view_rem[col])
        v_p = encoding_for(enc_plain, cols, col, tok_p)
        v_r = encoding_for(enc_rem, cols, col, tok_r)
        l1 = float(np.abs(v_p - v_r).sum())
        l1_by_col[col] = l1
        max_l1 = max(max_l1, l1)
        print(f"{col:<26} {'plain':<10} " + " ".join(f"{x:>10.4f}" for x in v_p)
              + f" {'':>12}  (token {tok_p!r})")
        print(f"{'':<26} {'remasked':<10} " + " ".join(f"{x:>10.4f}" for x in v_r)
              + f" {l1:>12.4f}")

    # the direct mechanism read: who lives in the nan level?
    print("\nclass posterior of the rows that CARRY the NaN (train) vs the")
    print("uniform-mask counterfactual population (= the train marginal):")
    print(f"{'column':<26} {'P(y|col NaN) train':<38} {'L1 vs prior':>12}")
    for col in shifted:
        m = ds.train[col].isna().to_numpy()
        post = np.array([(y[m] == k).mean() for k in range(len(CLASSES))])
        l1p = float(np.abs(post - prior).sum())
        print(f"{col:<26} " + " ".join(f"{x:>11.4f}" for x in post)
              + f" {l1p:>12.4f}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("PRE-REGISTERED BAR")
    print("=" * 78)
    ok_zero = zero_share >= BAR_ZERO_SUPPORT
    ok_te = max_l1 >= BAR_TE_L1
    print(f"  zero-train-support test rows : {zero_share:.2%} "
          f"(need >= {BAR_ZERO_SUPPORT:.0%})  -> {'PASS' if ok_zero else 'FAIL'}")
    print(f"  max-column nan-level TE L1   : {max_l1:.4f} "
          f"(need >= {BAR_TE_L1})       -> {'PASS' if ok_te else 'FAIL'}")
    print(f"  per-column L1: { {k: round(v, 4) for k, v in l1_by_col.items()} }")
    verdict = "PASS (both)" if (ok_zero and ok_te) else "FAIL"
    print(f"\nVERDICT E1.1: {verdict}")


if __name__ == "__main__":
    main()
