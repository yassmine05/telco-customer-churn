"""Tests for src/schema.py and its use in data.load_and_clean."""
import pandas as pd
import pytest

import config
from data import load_and_clean
from schema import (
    ALLOWED_LEVELS,
    INFERENCE_SCHEMA,
    RAW_SCHEMA,
    DataValidationError,
    validate,
)


def _parsed(raw_df):
    """The raw frame at the point load_and_clean validates it: TotalCharges numeric."""
    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def _load_corrupted(raw_df, tmp_path, corrupt):
    """Write a corrupted copy of the fixture to CSV and run load_and_clean on it."""
    bad = raw_df.copy()
    corrupt(bad)
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(DataValidationError) as excinfo:
        load_and_clean(path)
    return str(excinfo.value)


# ------------------------------------------------------------------ passing
def test_synthetic_fixture_passes_the_raw_schema(raw_df):
    validate(_parsed(raw_df), RAW_SCHEMA)


def test_fixture_features_pass_the_inference_schema(raw_df):
    validate(_parsed(raw_df).drop(columns=[config.TARGET, "customerID"]), INFERENCE_SCHEMA)


def test_allowed_levels_cover_every_categorical_feature():
    assert set(ALLOWED_LEVELS) == set(config.CATEGORICAL_FEATURES)


# ------------------------------------------------------------- rejections
def test_unknown_gender_is_rejected(raw_df, tmp_path):
    def corrupt(df):
        df.loc[5, "gender"] = "Unknown"
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert "column 'gender'" in message
    assert "'Unknown'" in message
    assert "rows [5]" in message


def test_unknown_payment_method_is_rejected(raw_df, tmp_path):
    def corrupt(df):
        df.loc[7, "PaymentMethod"] = "Bitcoin"
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert "column 'PaymentMethod'" in message
    assert "'Bitcoin'" in message


def test_negative_monthly_charges_are_rejected(raw_df, tmp_path):
    def corrupt(df):
        df.loc[3, "MonthlyCharges"] = -10.0
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert "column 'MonthlyCharges'" in message
    assert "greater_than(0)" in message


def test_duplicated_customer_id_is_rejected(raw_df, tmp_path):
    def corrupt(df):
        df.loc[1, "customerID"] = df.loc[0, "customerID"]
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert "column 'customerID'" in message
    assert "field_uniqueness" in message
    assert "rows [0, 1]" in message


def test_add_on_without_internet_is_rejected(raw_df, tmp_path):
    row = raw_df.index[raw_df["InternetService"] == "No"][0]

    def corrupt(df):
        df.loc[row, "OnlineSecurity"] = "Yes"
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert "InternetService" in message and "OnlineSecurity" in message
    assert f"rows [{row}]" in message


def test_multiple_lines_without_phone_is_rejected(raw_df):
    df = _parsed(raw_df)
    row = df.index[df["PhoneService"] == "No"][0]
    df.loc[row, "MultipleLines"] = "No"
    with pytest.raises(DataValidationError, match="MultipleLines"):
        validate(df, RAW_SCHEMA)


def test_zero_total_charges_with_positive_tenure_is_rejected(raw_df):
    df = _parsed(raw_df)
    row = df.index[df["tenure"] > 0][0]
    df.loc[row, "TotalCharges"] = 0.0
    with pytest.raises(DataValidationError, match="TotalCharges is 0 only when tenure == 0"):
        validate(df, RAW_SCHEMA)


def test_unknown_churn_label_is_rejected(raw_df):
    df = _parsed(raw_df)
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="column 'Churn'"):
        validate(df, RAW_SCHEMA)


def test_all_failures_are_reported_at_once(raw_df, tmp_path):
    def corrupt(df):
        df.loc[5, "gender"] = "Unknown"
        df.loc[7, "PaymentMethod"] = "Bitcoin"
        df.loc[3, "MonthlyCharges"] = -10.0
    message = _load_corrupted(raw_df, tmp_path, corrupt)
    assert message.startswith("raw Telco data: 3 validation problem(s)")
    for column in ("gender", "PaymentMethod", "MonthlyCharges"):
        assert f"column '{column}'" in message


def test_unexpected_column_is_rejected(raw_df):
    df = _parsed(raw_df).assign(extra=1)
    with pytest.raises(DataValidationError, match="unexpected column.*extra"):
        validate(df, RAW_SCHEMA)


# -------------------------------------------------------------- inference
def _one_customer(raw_df):
    return _parsed(raw_df).drop(columns=[config.TARGET, "customerID"]).head(1)


def test_inference_schema_rejects_unknown_gender(raw_df):
    # The same "Unknown" row that tests/test_train.py shows would be scored
    # silently as "Female" is blocked before it reaches predict_proba.
    customer = _one_customer(raw_df).assign(gender="Unknown")
    with pytest.raises(DataValidationError) as excinfo:
        validate(customer, INFERENCE_SCHEMA)
    assert "column 'gender'" in str(excinfo.value)


def test_inference_schema_accepts_an_optional_customer_id(raw_df):
    customer = _one_customer(raw_df).assign(customerID="0000-TEST")
    validate(customer, INFERENCE_SCHEMA)


def test_inference_schema_rejects_a_churn_column(raw_df):
    customer = _one_customer(raw_df).assign(Churn="No")
    with pytest.raises(DataValidationError, match="unexpected column.*Churn"):
        validate(customer, INFERENCE_SCHEMA)
