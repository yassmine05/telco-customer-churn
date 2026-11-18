"""Tests for src/train.py: pipelines, leakage, class weighting, reproducibility."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import config
from train import make_baseline, make_xgboost

from conftest import SMALL_XGB_PARAMS


# ------------------------------------------------- preprocessing / leakage
@pytest.mark.parametrize("factory", [make_baseline, lambda: make_xgboost(1.0)],
                         ids=["baseline", "xgboost"])
def test_preprocessing_is_the_first_step_of_the_pipeline(factory):
    # Inside a Pipeline, cross_val_predict and GridSearchCV re-fit the
    # preprocessing on each training fold, so validation rows never inform it.
    pipe = factory()
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["prep", "model"]
    assert isinstance(pipe.named_steps["prep"], ColumnTransformer)


def test_scaler_is_fitted_on_training_rows_only(split_data):
    X_train, X_test, y_train, _ = split_data
    pipe = make_baseline().fit(X_train, y_train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    train_mean = X_train[config.NUMERIC_FEATURES].mean().to_numpy()
    all_mean = (pd.concat([X_train, X_test])[config.NUMERIC_FEATURES]
                .mean().to_numpy())
    np.testing.assert_allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, all_mean)


def test_encoder_categories_come_from_the_rows_it_was_fitted_on(split_data):
    # Fit on rows without any two-year contracts: the encoder must not know
    # that category, even though it exists elsewhere in the data.
    X_train, X_test, y_train, _ = split_data
    keep = (X_train["Contract"] != "Two year").to_numpy()
    pipe = make_baseline().fit(X_train[keep], y_train[keep])
    encoder = pipe.named_steps["prep"].named_transformers_["cat"]
    contract_levels = encoder.categories_[config.CATEGORICAL_FEATURES.index("Contract")]
    assert "Two year" not in contract_levels
    # ...and scoring rows that do have it still works.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "unknown categories" notice
        proba = pipe.predict_proba(X_test)[:, 1]
    assert np.isfinite(proba).all()


def test_customer_id_never_reaches_the_model(fitted_xgb):
    names = fitted_xgb.named_steps["prep"].get_feature_names_out()
    assert not any("customerID" in n for n in names)


@pytest.mark.parametrize("model", ["baseline", "xgboost"])
def test_unseen_categories_do_not_crash_prediction(split_data, fitted_xgb, model):
    X_train, X_test, y_train, _ = split_data
    pipe = make_baseline().fit(X_train, y_train) if model == "baseline" else fitted_xgb
    odd = X_test.copy()
    odd["PaymentMethod"] = "Crypto wallet"
    odd["InternetService"] = "Satellite"
    odd["gender"] = "Unknown"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # sklearn warns, by design
        proba = pipe.predict_proba(odd)[:, 1]
    assert proba.shape == (len(odd),)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_unseen_value_in_a_yes_no_column_is_read_as_the_dropped_level(split_data,
                                                                      fitted_xgb):
    # Documents a quirk of drop="if_binary" + handle_unknown="ignore": an
    # unknown value in a two-level column encodes as all zeros, which is the
    # same as the dropped first level ("Female" for gender). It is silently
    # treated as that level rather than as "unknown".
    _, X_test, _, _ = split_data
    as_unknown, as_female = X_test.copy(), X_test.copy()
    as_unknown["gender"] = "Unknown"
    as_female["gender"] = "Female"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        np.testing.assert_array_equal(fitted_xgb.predict_proba(as_unknown),
                                      fitted_xgb.predict_proba(as_female))


# -------------------------------------------------------- class weighting
def test_make_xgboost_passes_scale_pos_weight_to_the_model():
    model = make_xgboost(2.5).named_steps["model"]
    assert model.get_params()["scale_pos_weight"] == 2.5


def test_scale_pos_weight_is_negatives_over_positives_of_the_training_set(pipeline_run):
    y_train = pipeline_run.y_train
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert pipeline_run.meta["scale_pos_weight"] == pytest.approx(expected)

    # Computed on training rows only: the full dataset gives a different ratio.
    y_all = np.concatenate([y_train, pipeline_run.y_test])
    assert expected != pytest.approx((y_all == 0).sum() / (y_all == 1).sum())

    import joblib
    saved = joblib.load(pipeline_run.root / "models" / "xgboost.joblib")
    assert saved.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)


# -------------------------------------------------------- reproducibility
def test_training_twice_gives_identical_xgboost_predictions(split_data):
    X_train, X_test, y_train, _ = split_data
    runs = [make_xgboost(2.7, **SMALL_XGB_PARAMS).fit(X_train, y_train)
            .predict_proba(X_test) for _ in range(2)]
    np.testing.assert_array_equal(runs[0], runs[1])


def test_training_twice_gives_identical_baseline_predictions(split_data):
    X_train, X_test, y_train, _ = split_data
    runs = [make_baseline().fit(X_train, y_train).predict_proba(X_test) for _ in range(2)]
    np.testing.assert_array_equal(runs[0], runs[1])


def test_xgboost_uses_the_project_seed():
    assert make_xgboost(1.0).named_steps["model"].get_params()["random_state"] == config.SEED
