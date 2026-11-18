"""Pandera schemas for the raw Telco data and for a single customer to score.

Why this exists: the preprocessing uses OneHotEncoder(drop="if_binary",
handle_unknown="ignore"). An unknown value in a two-level column (for example
gender="Unknown") then encodes as all zeros, which is the same as the dropped
level ("Female"), so the model silently scores it as that level. Validating
the input first rejects such rows before they reach the model.

Two schemas:

* RAW_SCHEMA: the Kaggle CSV after TotalCharges has been parsed to a number
  (blank -> 0 for tenure-0 customers), before customerID is dropped and Churn
  is mapped to 0/1. customerID and Churn are required.
* INFERENCE_SCHEMA: one or more customers to score. No Churn column;
  customerID is optional.

Allowed category levels are the ones in the real file (checked against all
7,043 rows), not guesses.
"""
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

import config

YES_NO = ["Yes", "No"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]

ALLOWED_LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{col: ["Yes", "No", "No internet service"] for col in ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}
if set(ALLOWED_LEVELS) != set(config.CATEGORICAL_FEATURES):
    raise RuntimeError("ALLOWED_LEVELS and config.CATEGORICAL_FEATURES list different columns")


class DataValidationError(ValueError):
    """Raised when data fails a schema. The message lists every failure."""


# ------------------------------------------------------------ cross-column checks
# Each returns a boolean Series (True = row is fine). The check names spell out
# the columns involved so the error message points at them.
def _addons_match_internet(df: pd.DataFrame) -> pd.Series:
    no_internet = df["InternetService"] == "No"
    addons_none = (df[ADDONS] == "No internet service").all(axis=1)
    addons_some = (df[ADDONS] != "No internet service").all(axis=1)
    return (no_internet & addons_none) | (~no_internet & addons_some)


def _multiple_lines_match_phone(df: pd.DataFrame) -> pd.Series:
    no_phone = df["PhoneService"] == "No"
    no_lines = df["MultipleLines"] == "No phone service"
    return no_phone == no_lines


def _zero_total_only_for_new_customers(df: pd.DataFrame) -> pd.Series:
    return (df["TotalCharges"] != 0) | (df["tenure"] == 0)


CROSS_CHECKS = [
    pa.Check(_addons_match_internet,
             name="InternetService/add-ons consistent",
             error="InternetService == 'No' if and only if all of " + ", ".join(ADDONS)
                   + " are 'No internet service'"),
    pa.Check(_multiple_lines_match_phone,
             name="PhoneService/MultipleLines consistent",
             error="PhoneService == 'No' if and only if MultipleLines == 'No phone service'"),
    pa.Check(_zero_total_only_for_new_customers,
             name="TotalCharges/tenure consistent",
             error="TotalCharges is 0 only when tenure == 0"),
]


def _feature_columns() -> dict:
    columns = {
        col: pa.Column(nullable=False, checks=pa.Check.isin(levels))
        for col, levels in ALLOWED_LEVELS.items()
    }
    columns.update({
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "tenure": pa.Column(int, pa.Check.in_range(0, 100), nullable=False),
        "MonthlyCharges": pa.Column(float, [pa.Check.gt(0), pa.Check.lt(500)],
                                    nullable=False),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=False),
    })
    return columns


def _make_schema(name: str, with_target: bool, id_required: bool) -> pa.DataFrameSchema:
    columns = {"customerID": pa.Column(nullable=False, unique=True, required=id_required),
               **_feature_columns()}
    if with_target:
        columns[config.TARGET] = pa.Column(checks=pa.Check.isin(YES_NO), nullable=False)
    # strict: an unexpected column is an error, because ColumnTransformer would
    # silently drop it.
    return pa.DataFrameSchema(columns, checks=CROSS_CHECKS, strict=True, name=name)


RAW_SCHEMA = _make_schema("raw Telco data", with_target=True, id_required=True)
INFERENCE_SCHEMA = _make_schema("customer to score", with_target=False, id_required=False)


# ------------------------------------------------------------------- validate
def _describe(errors: SchemaErrors, schema_name: str, max_examples: int = 5) -> str:
    cases = errors.failure_cases.copy()
    # Cross-column checks are reported by pandera once per column of every
    # failing row; collapse them to one line per check (the check text names
    # the columns involved).
    cross = cases["check"].isin([check.error for check in CROSS_CHECKS])
    cases.loc[cross, "column"] = "(cross-column)"
    cases.loc[cross, "failure_case"] = None
    cases["column"] = cases["column"].fillna("(table)")
    table_level = {"column_in_dataframe": "(missing column)",
                   "column_in_schema": "(unexpected column)"}
    for check, label in table_level.items():
        cases.loc[cases["check"] == check, "column"] = label
    lines = []
    for (column, check), group in cases.groupby(["column", "check"], sort=False,
                                                 dropna=False):
        rows = group["index"].dropna().astype(int).unique().tolist()
        values = group["failure_case"].dropna().astype(str).unique().tolist()
        if rows:
            where = f"{len(rows)} row(s), e.g. rows {rows[:max_examples]}"
        elif column == "(cross-column)":
            where = "could not be run (a column it needs is missing?)"
        else:
            where = "whole table"
        if column in table_level.values():
            shown = f"; {values}"  # list every missing / unexpected column
        elif values:
            shown = f"; values {values[:max_examples]}"
        else:
            shown = ""
        lines.append(f"  - column '{column}', check {check}: {where}{shown}")
    return (f"{schema_name}: {len(lines)} validation problem(s)\n" + "\n".join(lines))


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate every check (lazy) and raise one DataValidationError listing all
    failures. Returns the validated frame."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as errors:
        raise DataValidationError(_describe(errors, schema.name)) from errors
