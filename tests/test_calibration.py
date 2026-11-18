"""Tests for probability calibration: the calibrated XGBoost model in train.py
and the Brier / reliability helpers in metrics.py."""
import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
import metrics
from conftest import SMALL_XGB_PARAMS
from train import CALIBRATION_METHODS, compare_calibration, make_calibrated, make_xgboost

# Isotonic calibration maps ranges of scores to the same probability (ties),
# and the calibrated model averages 5 boosters fitted on 80% of the rows, so
# its ranking is close to, but not identical with, the single raw booster.
AUC_TOLERANCE = 0.03


@pytest.fixture(scope="module")
def cv():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


@pytest.fixture(scope="module")
def xgb_and_data(split_data):
    X_train, X_test, y_train, y_test = split_data
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    return make_xgboost(spw, **SMALL_XGB_PARAMS), X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def comparison(xgb_and_data, cv):
    xgb, X_train, _, y_train, _ = xgb_and_data
    raw_oof = cross_val_predict(xgb, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    method, scores, oof = compare_calibration(xgb, X_train, y_train, cv)
    return raw_oof, method, scores, oof


# ------------------------------------------------------------ Brier score
def test_brier_score_on_known_inputs():
    # (0 + 0 + 0.25 + 0.25) / 4
    assert metrics.brier_score([1, 0, 1, 0], [1.0, 0.0, 0.5, 0.5]) == pytest.approx(0.125)
    assert metrics.brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == 0.0
    assert metrics.brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_of_predicting_the_base_rate_is_r_times_one_minus_r():
    y = np.array([1] * 27 + [0] * 73)
    assert metrics.brier_score(y, np.full(100, 0.27)) == pytest.approx(0.27 * 0.73)


def test_brier_penalises_inflated_scores_with_the_same_ranking():
    # Same ordering, so ROC-AUC is unchanged, but pushing every probability up
    # (as scale_pos_weight does) makes the Brier score worse.
    rng = np.random.default_rng(0)
    p = rng.random(5_000)
    y = (rng.random(5_000) < p).astype(int)         # p is calibrated by construction
    inflated = np.sqrt(p)                           # monotone, but runs high
    assert roc_auc_score(y, inflated) == pytest.approx(roc_auc_score(y, p))
    assert metrics.brier_score(y, inflated) > metrics.brier_score(y, p)


# ------------------------------------------------------ calibration bins
def test_calibration_bins_on_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    bins = metrics.calibration_bins(y, p)
    assert list(bins.columns) == ["mean_predicted", "observed_rate"]
    assert len(bins) == 10
    np.testing.assert_allclose(bins["observed_rate"], bins["mean_predicted"], atol=0.02)


def test_calibration_bins_are_equal_count():
    # Quantile strategy: with 1,000 distinct scores, each bin holds 100 rows,
    # so bin k's mean prediction is the mean of the k-th block of sorted scores.
    p = np.linspace(0.001, 0.999, 1_000)
    y = np.zeros(1_000, dtype=int)
    y[::2] = 1
    bins = metrics.calibration_bins(y, p)
    expected = p.reshape(10, 100).mean(axis=1)
    np.testing.assert_allclose(bins["mean_predicted"], expected, atol=1e-9)


def test_bayes_threshold_is_one_sixth_with_5_to_1_costs():
    assert metrics.bayes_threshold() == pytest.approx(1 / 6)


def test_bayes_threshold_follows_the_costs_in_config(monkeypatch):
    monkeypatch.setattr(config, "COST_FALSE_NEGATIVE", 3.0)
    assert metrics.bayes_threshold() == pytest.approx(0.25)


# --------------------------------------------------- calibrated XGBoost
def test_make_calibrated_wraps_the_pipeline_with_the_project_cv(xgb_and_data, cv):
    xgb = xgb_and_data[0]
    for method in CALIBRATION_METHODS:
        cal = make_calibrated(xgb, method, cv)
        assert isinstance(cal, CalibratedClassifierCV)
        assert cal.method == method
        assert cal.cv is cv
        assert cal.estimator is xgb


def test_calibrated_probabilities_are_in_0_1(xgb_and_data, cv):
    xgb, X_train, X_test, y_train, _ = xgb_and_data
    for method in CALIBRATION_METHODS:
        proba = make_calibrated(xgb, method, cv).fit(X_train, y_train).predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_calibration_keeps_the_ranking_on_held_out_rows(xgb_and_data, cv):
    xgb, X_train, X_test, y_train, y_test = xgb_and_data
    raw = xgb.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    for method in CALIBRATION_METHODS:
        cal = make_calibrated(xgb, method, cv).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        assert roc_auc_score(y_test, cal) == pytest.approx(roc_auc_score(y_test, raw),
                                                           abs=AUC_TOLERANCE), method


def test_calibration_keeps_the_ranking_out_of_fold(comparison, xgb_and_data):
    raw_oof, _, _, oof = comparison
    y_train = xgb_and_data[3]
    for method, proba in oof.items():
        assert roc_auc_score(y_train, proba) == pytest.approx(roc_auc_score(y_train, raw_oof),
                                                              abs=AUC_TOLERANCE), method


def test_calibration_lowers_out_of_fold_brier_score(comparison, xgb_and_data):
    # Deterministic (seeded); on this fixture the raw scores run high (mean
    # ~0.36 vs a 0.27 churn rate) and both methods pull them back.
    raw_oof, method, scores, _ = comparison
    y_train = xgb_and_data[3]
    raw_brier = metrics.brier_score(y_train, raw_oof)
    assert scores[method]["brier"] <= raw_brier
    assert raw_oof.mean() > y_train.mean()


def test_the_chosen_method_has_the_lowest_out_of_fold_brier(comparison):
    _, method, scores, _ = comparison
    assert set(scores) == set(CALIBRATION_METHODS)
    assert scores[method]["brier"] == min(s["brier"] for s in scores.values())
    for s in scores.values():
        assert s["log_loss"] > 0


def test_compare_calibration_scores_are_out_of_fold(comparison, xgb_and_data, cv):
    # The scores must come from cross_val_predict, not from a model that was
    # fitted on the same rows it is scored on.
    xgb, X_train, _, y_train, _ = xgb_and_data
    _, method, _, oof = comparison
    in_sample = (make_calibrated(xgb, method, cv).fit(X_train, y_train)
                 .predict_proba(X_train)[:, 1])
    assert not np.allclose(oof[method], in_sample)


# ------------------------------------------------- full pipeline outputs
def test_calibration_choice_is_recorded_in_metadata(pipeline_run):
    cal = pipeline_run.meta["calibration"]
    assert cal["method"] in CALIBRATION_METHODS
    assert set(cal["oof_scores"]) == {"uncalibrated", *CALIBRATION_METHODS}
    chosen = cal["oof_scores"][cal["method"]]["brier"]
    assert chosen == min(cal["oof_scores"][m]["brier"] for m in CALIBRATION_METHODS)
    assert cal["theoretical_threshold"] == pytest.approx(1 / 6)
    assert cal["empirical_threshold"] == pipeline_run.meta["thresholds"]["xgboost_calibrated"]


def test_saved_calibrated_model_uses_the_chosen_method(pipeline_run):
    model = joblib.load(pipeline_run.root / "models" / "xgboost_calibrated.joblib")
    assert isinstance(model, CalibratedClassifierCV)
    assert model.method == pipeline_run.meta["calibration"]["method"]
    assert len(model.calibrated_classifiers_) == config.CV_FOLDS


def test_shap_still_explains_the_uncalibrated_booster(pipeline_run):
    raw = joblib.load(pipeline_run.root / "models" / "xgboost.joblib")
    assert [name for name, _ in raw.steps] == ["prep", "model"]
