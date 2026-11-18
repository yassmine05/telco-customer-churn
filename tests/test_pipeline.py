"""End-to-end smoke test and leakage checks on the full eda -> train ->
evaluate -> explain run (see the pipeline_run fixture in conftest.py)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from train import make_calibrated, make_xgboost

EXPECTED_FIGURES = [
    "01_churn_by_contract.png", "02_churn_by_tenure.png",
    "03_churn_by_payment_method.png", "04_churn_by_internet_service.png",
    "05_threshold_cost_curve.png", "06_confusion_matrices.png",
    "07_precision_recall_curve.png", "08_gains_chart.png",
    "09_shap_summary.png", "10_shap_waterfall_customer.png",
    "11_calibration_curve.png",
]
MODELS = ("baseline_logreg", "xgboost", "xgboost_calibrated")


# ------------------------------------------------------------------ smoke
def test_all_figures_are_written(pipeline_run):
    figures = pipeline_run.root / "reports" / "figures"
    for name in EXPECTED_FIGURES:
        assert (figures / name).stat().st_size > 0, name


def test_reports_models_and_results_are_written(pipeline_run):
    root = pipeline_run.root
    for rel in ["reports/eda_insights.md", "models/baseline_logreg.joblib",
                "models/xgboost.joblib", "models/xgboost_calibrated.joblib",
                "models/metadata.json",
                "results/xgb_cv_results.csv", "results/nested_cv_results.csv",
                "results/shap_importance.csv",
                "results/ranked_customers_test.csv", "results/metrics.csv"]:
        assert (root / rel).stat().st_size > 0, rel


def test_metrics_csv_has_one_row_per_model_rule_and_dataset(pipeline_run):
    df = pd.read_csv(pipeline_run.root / "results" / "metrics.csv")
    assert len(df) == 2 * len(MODELS) * 2
    assert set(df["data"]) == {"train_cv_out_of_fold", "test"}
    assert set(df["model"]) == set(MODELS)
    assert set(df["threshold_rule"]) == {"default_0.5", "min_cost"}
    for col in ["precision", "recall", "roc_auc", "pr_auc", "brier"]:
        assert df[col].between(0, 1).all(), col


def test_grid_search_table_covers_every_configuration(pipeline_run):
    cv = pd.read_csv(pipeline_run.root / "results" / "xgb_cv_results.csv")
    assert len(cv) == 2                       # TINY_GRID has 2 combinations
    assert cv["rank_test_pr_auc"].iloc[0] == 1
    assert pipeline_run.meta["xgb_best_params"]["max_depth"] in (2, 3)


def test_nested_cv_outputs_are_written(pipeline_run):
    nested = pd.read_csv(pipeline_run.root / "results" / "nested_cv_results.csv")
    assert len(nested) == 2 * config.OUTER_CV_FOLDS
    summary = pipeline_run.meta["nested_cv"]
    for model in ("baseline_logreg", "xgboost"):
        assert set(summary[model]) == {"roc_auc_mean", "roc_auc_std",
                                       "pr_auc_mean", "pr_auc_std"}
    assert len(summary["xgb_params_by_fold"]) == config.OUTER_CV_FOLDS


def test_ranked_list_covers_every_test_customer_riskiest_first(pipeline_run):
    ranked = pd.read_csv(pipeline_run.root / "results" / "ranked_customers_test.csv",
                         index_col="row_id")
    assert sorted(ranked.index) == sorted(pipeline_run.X_test.index)
    assert list(ranked["risk_rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["churn_probability"].is_monotonic_decreasing
    # The list is the calibrated model's probability at its own threshold.
    calibrated = joblib.load(pipeline_run.root / "models" / "xgboost_calibrated.joblib")
    expected = pd.Series(calibrated.predict_proba(pipeline_run.X_test)[:, 1],
                         index=pipeline_run.X_test.index).round(4)
    np.testing.assert_allclose(ranked["churn_probability"], expected[ranked.index], atol=1e-4)
    threshold = pipeline_run.meta["thresholds"]["xgboost_calibrated"]
    assert (ranked["flagged_for_offer"] == (ranked["churn_probability"] >= threshold)).all()


def test_real_project_outputs_are_not_touched(pipeline_run):
    assert pipeline_run.guarded_digests_after == pipeline_run.guarded_digests_before


# ------------------------------------------------- test set stays held out
def test_eda_train_and_explain_ran_without_reading_held_out_data(pipeline_run):
    # eda.py and train.py were given Untouchable objects instead of the test
    # split, and explain.py instead of the test labels. Any read raises, so
    # reaching this point means they never used them.
    assert pipeline_run.meta["thresholds"]


# ------------------------------------------------- threshold selection
def test_thresholds_are_chosen_on_training_rows_only(pipeline_run):
    calls = pipeline_run.threshold_calls
    assert len(calls) == len(MODELS)          # baseline, XGBoost, calibrated XGBoost
    for call in calls:
        assert call["y"].index.equals(pipeline_run.y_train.index)
        assert call["y"].index.intersection(pipeline_run.X_test.index).empty
        assert len(call["proba"]) == len(pipeline_run.X_train)


def test_thresholds_are_chosen_from_out_of_fold_predictions(pipeline_run):
    # Recompute XGBoost's out-of-fold predictions independently: they must be
    # exactly the scores the threshold was chosen from.
    X_train, y_train, meta = pipeline_run.X_train, pipeline_run.y_train, pipeline_run.meta
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    model = make_xgboost(meta["scale_pos_weight"], **meta["xgb_best_params"])
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    np.testing.assert_allclose(pipeline_run.threshold_calls[1]["proba"], oof, rtol=0, atol=1e-7)

    # ...and they are not the (over-confident) in-sample scores of the final model.
    in_sample = joblib.load(pipeline_run.root / "models" / "xgboost.joblib") \
        .predict_proba(X_train)[:, 1]
    assert not np.allclose(oof, in_sample)


def test_calibrated_threshold_is_chosen_from_calibrated_out_of_fold_predictions(pipeline_run):
    # Recompute the calibrated model's out-of-fold probabilities (outer CV
    # around a calibrator with its own inner CV) and compare.
    X_train, y_train, meta = pipeline_run.X_train, pipeline_run.y_train, pipeline_run.meta
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    xgb = make_xgboost(meta["scale_pos_weight"], **meta["xgb_best_params"])
    model = make_calibrated(xgb, meta["calibration"]["method"], cv)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    call = pipeline_run.threshold_calls[2]
    np.testing.assert_allclose(call["proba"], oof, rtol=0, atol=1e-7)
    assert call["y"].index.equals(y_train.index)

    # Not the final calibrated model's in-sample scores, and not test-set scores.
    final = joblib.load(pipeline_run.root / "models" / "xgboost_calibrated.joblib")
    assert not np.allclose(oof, final.predict_proba(X_train)[:, 1])
    assert len(call["proba"]) != len(pipeline_run.X_test)


def test_saved_thresholds_are_the_ones_selected_in_training(pipeline_run):
    chosen = [c["threshold"] for c in pipeline_run.threshold_calls]
    assert [pipeline_run.meta["thresholds"][m] for m in MODELS] == chosen


def test_evaluation_uses_the_frozen_thresholds(pipeline_run):
    df = pd.read_csv(pipeline_run.root / "results" / "metrics.csv")
    test_rows = df[(df["data"] == "test") & (df["threshold_rule"] == "min_cost")]
    for _, row in test_rows.iterrows():
        assert row["threshold"] == pytest.approx(pipeline_run.meta["thresholds"][row["model"]])
