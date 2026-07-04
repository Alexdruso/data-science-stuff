"""Tests for data_science_stuff.kaggle.encoding."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data_science_stuff.kaggle.encoding import (
    add_fold_safe_target_encoding,
    add_frequency_features,
    add_quantile_bin_features,
    cat_key,
    qcut_codes,
    select_low_cardinality_cols,
    sorted_factorize,
    te_source_columns,
)


def test_cat_key_stringifies_and_fills_missing() -> None:
    keyed = cat_key(pd.Series(["a", None, 3]))
    assert keyed.tolist() == ["a", "__NA__", "3"]


def test_sorted_factorize_shares_codes_across_splits() -> None:
    tr = pd.Series(["b", "a", "c"])
    va = pd.Series(["a", "c"])
    te = pd.Series(["c", "d"])  # "d" only appears in test

    codes_tr, codes_va, codes_te = sorted_factorize(tr, va, te)

    # Codes follow the sorted union a<b<c<d and agree across splits.
    assert codes_tr.tolist() == [1, 0, 2]
    assert codes_va.tolist() == [0, 2]
    assert codes_te.tolist() == [2, 3]
    assert codes_tr.dtype == np.int32


def test_te_source_columns_matches_prefix() -> None:
    selected = ["u_g", "TE_spectral_GALAXY", "TE_qbin64_QSO", "redshift"]
    cat_cols = ["spectral", "qbin64", "unused_cat"]
    assert te_source_columns(selected, cat_cols) == ["spectral", "qbin64"]


def _te_fixture() -> "tuple[pd.DataFrame, NDArray[np.int64], pd.DataFrame]":
    """Category 'a' rows are class 1, 'b' rows class 0 — a perfect predictor."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).astype(np.int64)
    x_tr = pd.DataFrame({"cat": np.where(y == 1, "a", "b"), "num": rng.random(n)})
    x_va = pd.DataFrame({"cat": ["a", "b", "unseen"], "num": [0.1, 0.2, 0.3]})
    return x_tr, y, x_va


def test_add_fold_safe_target_encoding_adds_columns_everywhere() -> None:
    x_tr, y, x_va = _te_fixture()

    enc_tr, enc_va = add_fold_safe_target_encoding(
        x_tr, y, [x_va], ["cat"], {0: "NEG", 1: "POS"}
    )

    for frame in (enc_tr, enc_va):
        assert "TE_cat_NEG" in frame.columns
        assert "TE_cat_POS" in frame.columns
        assert frame["TE_cat_POS"].dtype == np.float32
    # Inputs are not mutated.
    assert "TE_cat_POS" not in x_tr.columns
    assert "TE_cat_POS" not in x_va.columns


def test_add_fold_safe_target_encoding_orders_categories_correctly() -> None:
    x_tr, y, x_va = _te_fixture()

    _, enc_va = add_fold_safe_target_encoding(
        x_tr, y, [x_va], ["cat"], {0: "NEG", 1: "POS"}, smooth=1.0
    )

    te_pos = enc_va["TE_cat_POS"].to_numpy()
    # 'a' rows are (almost) all class 1, 'b' rows class 0.
    assert te_pos[0] > 0.9
    assert te_pos[1] < 0.1
    # An unseen category falls back near the global prior, far from either extreme.
    assert 0.2 < te_pos[2] < 0.8


def test_add_fold_safe_target_encoding_shuffled_targets_stay_near_prior() -> None:
    x_tr, y, x_va = _te_fixture()
    rng = np.random.default_rng(1)
    y_shuffled = rng.permutation(y)
    prior = float(y_shuffled.mean())

    _, enc_va = add_fold_safe_target_encoding(
        x_tr, y_shuffled, [x_va], ["cat"], {1: "POS"}
    )

    # With targets decoupled from the category, encodings hug the prior —
    # the signature of an encoder that is not leaking.
    np.testing.assert_allclose(enc_va["TE_cat_POS"].to_numpy(), prior, atol=0.1)


def test_qcut_codes_monotonic_and_reference_driven() -> None:
    ref = np.arange(100, dtype=np.float32)
    values = np.array([0.0, 25.0, 50.0, 99.0], dtype=np.float32)
    codes = qcut_codes(values, ref, 4)
    assert codes.tolist() == [0, 1, 2, 3]
    assert codes.dtype == np.int16


def test_qcut_codes_edge_cases_code_to_minus_one() -> None:
    ref = np.arange(10, dtype=np.float32)
    values = np.array([-5.0, 5.0, 50.0, np.nan], dtype=np.float32)
    codes = qcut_codes(values, ref, 2)
    assert codes[0] == -1  # below range
    assert codes[2] == -1  # above range
    assert codes[3] == -1  # NaN
    # Constant reference → no usable edges.
    assert (qcut_codes(values, np.ones(10, np.float32), 4) == -1).all()
    # Too-small reference → no usable edges.
    assert (qcut_codes(values, np.array([1.0], np.float32), 4) == -1).all()


def test_add_quantile_bin_features_names_and_crosses() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"u": rng.random(50), "g": rng.random(50)})
    fit_mask = np.zeros(50, dtype=bool)
    fit_mask[:40] = True  # only "train" rows drive the bin edges

    out, new_cols = add_quantile_bin_features(
        df,
        fit_mask,
        cols=["u", "g", "absent"],
        quantiles=(4, 8),
        crosses=[("u_qbin4", "g_qbin4")],
    )

    assert new_cols == [
        "u_qbin4",
        "u_qbin8",
        "g_qbin4",
        "g_qbin8",
        "u_qbin4__x__g_qbin4",
    ]
    assert all(c in out.columns for c in new_cols)
    assert out["u_qbin4__x__g_qbin4"].str.contains("__").all()
    assert "absent_qbin4" not in out.columns


def test_select_low_cardinality_cols_caps_cardinality() -> None:
    df = pd.DataFrame({"low": ["a", "b"] * 50, "high": [str(i) for i in range(100)]})
    assert select_low_cardinality_cols(df, ["low", "high", "absent"], 10) == ["low"]


def test_add_frequency_features_counts_only_masked_rows() -> None:
    df = pd.DataFrame({"cat": ["a", "a", "b", "a", "b"]})
    fit_mask = np.array([True, True, True, False, False])

    out = add_frequency_features(df, ["cat"], fit_mask)

    # Counts come from the first three rows only: a=2, b=1.
    assert out["cat_freq"].tolist() == [2.0, 2.0, 1.0, 2.0, 1.0]
    np.testing.assert_allclose(
        out["cat_freq_log1p"].to_numpy(), np.log1p(out["cat_freq"].to_numpy())
    )
