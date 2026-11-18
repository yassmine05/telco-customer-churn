"""Tests for src/explain.py: SHAP values are complete and add up."""
import numpy as np
import pandas as pd

from explain import explain


def _encoded(pipeline, X):
    prep = pipeline.named_steps["prep"]
    return pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())


def test_one_shap_value_per_feature_per_row(fitted_xgb, split_data):
    _, X_test, _, _ = split_data
    explanation = explain(fitted_xgb, X_test)
    names = fitted_xgb.named_steps["prep"].get_feature_names_out()
    assert explanation.values.shape == (len(X_test), len(names))
    assert list(explanation.feature_names) == list(names)


def test_base_value_plus_shap_reconstructs_raw_model_output(fitted_xgb, split_data):
    _, X_test, _, _ = split_data
    explanation = explain(fitted_xgb, X_test)
    margin = fitted_xgb.named_steps["model"].predict(_encoded(fitted_xgb, X_test),
                                                     output_margin=True)
    rebuilt = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(rebuilt, margin, atol=1e-4, rtol=0)


def test_reconstruction_matches_predicted_probability(fitted_xgb, split_data):
    _, X_test, _, _ = split_data
    explanation = explain(fitted_xgb, X_test)
    rebuilt = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(1 / (1 + np.exp(-rebuilt)),
                               fitted_xgb.predict_proba(X_test)[:, 1], atol=1e-5)


def test_single_customer_can_be_explained(fitted_xgb, split_data):
    # The Streamlit app explains one row at a time.
    _, X_test, _, _ = split_data
    explanation = explain(fitted_xgb, X_test.iloc[[0]])
    assert explanation.values.shape[0] == 1
