"""Tests for src/data.py: cleaning and the stratified train/test split."""
import numpy as np
import pandas as pd
import pytest

import config
from data import load_and_clean, split


# ------------------------------------------------------------------ cleaning
def test_fixture_has_blank_total_charges_stored_as_text(raw_df):
    # Precondition: the fixture reproduces the quirk the cleaning step handles.
    assert raw_df["TotalCharges"].map(type).eq(str).all()
    assert (raw_df["TotalCharges"] == " ").sum() == 6


def test_blank_total_charges_become_zero_for_new_customers(raw_df, clean_df):
    blank = (raw_df["TotalCharges"] == " ").to_numpy()
    assert pd.api.types.is_float_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].notna().all()
    assert (clean_df.loc[blank, "TotalCharges"] == 0.0).all()
    assert (clean_df.loc[blank, "tenure"] == 0).all()


def test_non_blank_total_charges_are_parsed_unchanged(raw_df, clean_df):
    filled = (raw_df["TotalCharges"] != " ").to_numpy()
    expected = raw_df.loc[filled, "TotalCharges"].astype(float).to_numpy()
    np.testing.assert_array_equal(clean_df.loc[filled, "TotalCharges"].to_numpy(), expected)


def test_blank_total_charges_with_positive_tenure_is_rejected(raw_df, tmp_path):
    # data.py only fills blanks with 0 because they belong to tenure-0 customers.
    # A blank for an existing customer would be a different problem, so it must
    # fail with a ValueError (an `assert` would vanish under `python -O`).
    bad = raw_df.copy()
    rows = bad.index[bad["tenure"] > 0][:2].tolist()
    bad.loc[rows, "TotalCharges"] = " "
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError) as excinfo:
        load_and_clean(path)
    message = str(excinfo.value)
    assert message.startswith("2 row(s) have a blank TotalCharges but tenure > 0")
    assert str(rows) in message


def test_load_and_clean_has_no_bare_asserts():
    # `python -O` strips assert statements, so data checks must not rely on them.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(load_and_clean)))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def test_churn_is_mapped_to_0_and_1(raw_df, clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    expected = (raw_df["Churn"] == "Yes").astype(int).to_numpy()
    np.testing.assert_array_equal(clean_df[config.TARGET].to_numpy(), expected)
    assert clean_df[config.TARGET].mean() == pytest.approx(0.27)


def test_customer_id_is_dropped(clean_df):
    assert "customerID" not in clean_df.columns


def test_every_cleaned_column_is_a_declared_feature(clean_df):
    # ColumnTransformer silently drops undeclared columns, so check none are missed.
    declared = set(config.NUMERIC_FEATURES + config.PASSTHROUGH_FEATURES
                   + config.CATEGORICAL_FEATURES)
    assert set(clean_df.columns) - {config.TARGET} == declared


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_and_clean(tmp_path / "nope.csv")


# --------------------------------------------------------------------- split
def test_split_is_80_20(split_data, clean_df):
    X_train, X_test, y_train, y_test = split_data
    assert len(X_train) == len(y_train) == 160
    assert len(X_test) == len(y_test) == 40
    assert config.TARGET not in X_train.columns


def test_split_keeps_churn_rate_within_two_points(split_data, clean_df):
    _, _, y_train, y_test = split_data
    overall = clean_df[config.TARGET].mean()
    assert abs(y_train.mean() - overall) <= 0.02
    assert abs(y_test.mean() - overall) <= 0.02


def test_no_customer_in_both_train_and_test(split_data, raw_df):
    X_train, X_test, _, _ = split_data
    train_ids = set(raw_df.loc[X_train.index, "customerID"])
    test_ids = set(raw_df.loc[X_test.index, "customerID"])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == len(raw_df)


def test_same_seed_gives_same_split(clean_df):
    first, second = split(clean_df), split(clean_df)
    for a, b in zip(first, second, strict=True):
        assert a.index.equals(b.index)


def test_split_depends_on_the_seed(clean_df, monkeypatch):
    # Shows the seed in config is what fixes the split (not an accident of order).
    default_test = split(clean_df)[1].index
    monkeypatch.setattr(config, "SEED", 7)
    assert not split(clean_df)[1].index.equals(default_test)
