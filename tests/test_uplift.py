"""Tests for the uplift module (src/uplift/) on synthetic customers.

Tests that compare model scores use LARGE_N synthetic rows and STRONG_EFFECTS,
not the shared 200-row fixture: with a few dozen rows per segment and arm, the
sign of an estimated uplift can flip with the package build or the seed (the
lesson of the calibration test that failed only on Python 3.11). The margins
quoted below come from a sweep over fixture seeds 2024-2043 x campaign seeds
0-4 (100 combinations); every assertion passed in all of them.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
import metrics
from conftest import SMALL_XGB_PARAMS, make_raw_telco
from data import load_and_clean, split
from train import make_calibrated, make_xgboost
from uplift.models import TLearner
from uplift.simulate import (
    SEGMENTS,
    assign_segments,
    control_probabilities,
    segment_summary,
    simulate,
    simulate_campaign,
)

LARGE_N = 2_000
# Qini checks need no model fit, so they use more rows to keep noise low.
QINI_N = 6_000
LARGE_SEED = 2024
CAMPAIGN_SEED = 0
# Much larger than the published effects in config.py, so the tests check the
# method, not whether a 1,200-row sample can resolve a 5-point effect.
STRONG_EFFECTS = {"sure_thing": 0.0, "sleeping_dog": -0.4, "lost_cause": 0.0,
                  "persuadable": 0.45, "other": 0.05}
STRONG_P_CONTROL = 0.5
# The persuadable rule is a three-way interaction (contract x tenure x tech
# support); the shared 30-tree setting underfits it (worst persuadable mean
# 0.08 over the seed sweep), 100 trees of depth 4 do not (worst 0.13).
TLEARNER_PARAMS = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100}


def load_large(tmp_path, fixture_seed=LARGE_SEED):
    """X_train, X_test, y_train, y_test from LARGE_N synthetic customers."""
    path = tmp_path / f"large_{fixture_seed}.csv"
    make_raw_telco(n=LARGE_N, n_churners=int(LARGE_N * 0.27),
                   seed=fixture_seed).to_csv(path, index=False)
    return split(load_and_clean(path))


def fit_tlearner_on_strong_campaign(X_train, X_test, campaign_seed=CAMPAIGN_SEED):
    """Simulate a strong-effect campaign with a fixed control probability, fit
    the T-learner on X_train, return (campaign on X_test, predicted uplift).

    X_test should be a separate, large sample: with 400 held-out rows one
    fixture seed had only 26 persuadables, and their mean was noisy."""
    train_sim = simulate_campaign(X_train, np.full(len(X_train), STRONG_P_CONTROL), campaign_seed,
                                  STRONG_EFFECTS)
    test_sim = simulate_campaign(X_test, np.full(len(X_test), STRONG_P_CONTROL), campaign_seed + 1,
                                 STRONG_EFFECTS)
    model = TLearner(**TLEARNER_PARAMS).fit(X_train, train_sim["treatment"],
                                            train_sim["churn"])
    return test_sim, model.predict_uplift(X_test)


def strong_campaign_qini(fixture_seed=LARGE_SEED, campaign_seed=CAMPAIGN_SEED, n_random=20):
    """Qini of the true-uplift ordering, and of n_random random orderings, on
    QINI_N synthetic customers (raw columns are enough to assign segments)."""
    X = make_raw_telco(n=QINI_N, n_churners=int(QINI_N * 0.27), seed=fixture_seed)
    rng = np.random.default_rng(campaign_seed)
    sim = simulate_campaign(X, rng.uniform(0.3, 0.6, len(X)), campaign_seed,
                            STRONG_EFFECTS)
    q_true = metrics.qini_coefficient(sim["churn"], sim["treatment"], sim["true_uplift"])
    q_random = [metrics.qini_coefficient(sim["churn"], sim["treatment"], rng.random(len(X)))
                for _ in range(n_random)]
    return q_true, np.array(q_random)


@pytest.fixture(scope="module")
def large_data(tmp_path_factory):
    return load_large(tmp_path_factory.mktemp("uplift"))


@pytest.fixture(scope="module")
def eval_customers(tmp_path_factory):
    """LARGE_N fresh customers (another seed), used only to score a T-learner."""
    X_a, X_b, _, _ = load_large(tmp_path_factory.mktemp("uplift_eval"), LARGE_SEED + 1000)
    return pd.concat([X_a, X_b])


@pytest.fixture(scope="module")
def calibrated(split_data):
    X_train, _, y_train, _ = split_data
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    return make_calibrated(make_xgboost(spw, **SMALL_XGB_PARAMS), "sigmoid", cv).fit(
        X_train, y_train)


# ------------------------------------------------------------- segments
def _customer(**overrides):
    base = {"Contract": "Month-to-month", "tenure": 30, "InternetService": "DSL",
            "PaymentMethod": "Mailed check", "TechSupport": "No"}
    return {**base, **overrides}


def test_segment_rules_on_hand_built_customers():
    X = pd.DataFrame([
        _customer(Contract="Two year", tenure=70),                         # sure thing
        _customer(tenure=60),                                              # sleeping dog
        _customer(tenure=60, InternetService="Fiber optic",
                  PaymentMethod="Electronic check"),                       # sleeping dog wins
        _customer(InternetService="Fiber optic", PaymentMethod="Electronic check"),
        _customer(tenure=5),                                               # persuadable
        _customer(tenure=5, TechSupport="Yes"),                            # other
        _customer(Contract="One year", tenure=5),                          # other
    ])
    assert assign_segments(X).tolist() == [
        "sure_thing", "sleeping_dog", "sleeping_dog", "lost_cause", "persuadable",
        "other", "other"]


def test_all_segments_present_in_synthetic_data(clean_df):
    assert set(assign_segments(clean_df)) == set(SEGMENTS)


# ----------------------------------------------------------- simulation
def test_simulation_is_reproducible_with_the_same_seed(clean_df):
    p0 = np.linspace(0.05, 0.9, len(clean_df))
    a = simulate_campaign(clean_df, p0, seed=3)
    b = simulate_campaign(clean_df, p0, seed=3)
    c = simulate_campaign(clean_df, p0, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not np.array_equal(a["treatment"], c["treatment"])


def test_treatment_is_about_half_and_random():
    X = make_raw_telco(n=QINI_N, n_churners=int(QINI_N * 0.27), seed=LARGE_SEED)
    sim = simulate_campaign(X, np.full(len(X), 0.3), seed=1)
    assert abs(sim["treatment"].mean() - 0.5) <= 1 / len(sim)
    # Random assignment: every segment is close to half treated too.
    by_segment = sim.groupby("segment")["treatment"].mean()
    assert by_segment.between(0.35, 0.65).all(), by_segment


def test_true_effect_follows_the_segment_and_is_clipped(clean_df):
    p0 = np.full(len(clean_df), 0.5)
    sim = simulate_campaign(clean_df, p0, seed=0, effects=STRONG_EFFECTS)
    for segment, tau in STRONG_EFFECTS.items():
        rows = sim["segment"] == segment
        np.testing.assert_allclose(sim.loc[rows, "true_uplift"], tau)
    # p0 = 0.1 with a 0.45 effect: P(churn | offer) is clipped at 0, so the
    # customer's true effect is 0.1, not 0.45.
    low = simulate_campaign(clean_df, np.full(len(clean_df), 0.1), seed=0,
                            effects=STRONG_EFFECTS)
    pers = low["segment"] == "persuadable"
    assert (low.loc[pers, "p_treated"] == 0).all()
    np.testing.assert_allclose(low.loc[pers, "true_uplift"], 0.1)
    assert low["p_treated"].between(0, 1).all()


def test_outcome_is_drawn_from_the_assigned_arm(large_data):
    X_train = large_data[0]
    sim = simulate_campaign(X_train, np.full(len(X_train), STRONG_P_CONTROL), seed=2,
                            effects=STRONG_EFFECTS)
    rates = sim.groupby(["segment", "treatment"])["churn"].mean()
    assert rates["persuadable", 0] - rates["persuadable", 1] > 0.2     # true 0.45
    assert rates["sleeping_dog", 1] - rates["sleeping_dog", 0] > 0.15  # true 0.40


def test_segment_summary_reports_size_and_true_effect(clean_df):
    sim = simulate_campaign(clean_df, np.full(len(clean_df), 0.5), seed=0,
                            effects=STRONG_EFFECTS)
    table = segment_summary(sim)
    assert table.loc["all", "customers"] == len(sim)
    assert table.loc[SEGMENTS, "customers"].sum() == len(sim)
    assert table.loc["persuadable", "mean_true_uplift"] == pytest.approx(0.45)
    assert table.loc["sleeping_dog", "mean_true_uplift"] == pytest.approx(-0.4)


# ------------------------------------------- control-arm probabilities
@pytest.fixture(scope="module")
def control_probs(calibrated, split_data):
    """control_probabilities() on the shared fixture, computed once (it refits
    the calibrated model CV_FOLDS times)."""
    X_train, X_test, y_train, _ = split_data
    return control_probabilities(calibrated, X_train, y_train, X_test)


def test_training_control_probabilities_are_out_of_fold(calibrated, split_data,
                                                        control_probs):
    X_train, X_test, y_train, _ = split_data
    p_train, p_test = control_probs
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    expected_oof = cross_val_predict(clone(calibrated), X_train, y_train, cv=cv,
                                     method="predict_proba")[:, 1]
    in_sample = calibrated.predict_proba(X_train)[:, 1]
    np.testing.assert_allclose(p_train, expected_oof)
    assert not np.allclose(p_train, in_sample)
    np.testing.assert_allclose(p_test, calibrated.predict_proba(X_test)[:, 1])


def test_simulate_uses_the_control_probabilities(calibrated, split_data, control_probs):
    X_train, X_test, y_train, _ = split_data
    train_sim, test_sim = simulate(calibrated, X_train, y_train, X_test, seed=5)
    np.testing.assert_allclose(train_sim["p_control"], control_probs[0])
    np.testing.assert_allclose(test_sim["p_control"], control_probs[1])
    assert train_sim.index.equals(X_train.index) and test_sim.index.equals(X_test.index)


# ------------------------------------------------------------ T-learner
def test_tlearner_uplift_is_control_minus_treated(large_data):
    X_train, X_test = large_data[0], large_data[1]
    sim = simulate_campaign(X_train, np.full(len(X_train), STRONG_P_CONTROL), 0,
                            STRONG_EFFECTS)
    model = TLearner(**SMALL_XGB_PARAMS).fit(X_train, sim["treatment"], sim["churn"])
    p_control, p_treated = model.predict_arms(X_test)
    np.testing.assert_allclose(model.predict_uplift(X_test), p_control - p_treated)
    # Each arm's model saw only its own customers, without class re-weighting.
    for arm in (0, 1):
        booster = model.models_[arm].named_steps["model"]
        assert booster.get_params()["scale_pos_weight"] == 1.0
        assert model.models_[arm].named_steps["prep"].n_features_in_ == X_train.shape[1]


def test_tlearner_recovers_the_sign_of_strong_effects(large_data, eval_customers):
    # Over the 100-seed sweep: persuadables' mean predicted uplift was 0.13 to
    # 0.46 (true 0.45), sleeping dogs' -0.46 to -0.25 (true -0.40).
    test_sim, predicted = fit_tlearner_on_strong_campaign(large_data[0], eval_customers)
    mean_by_segment = pd.Series(predicted, index=test_sim.index).groupby(
        test_sim["segment"]).mean()
    assert mean_by_segment["persuadable"] > 0.05, mean_by_segment
    assert mean_by_segment["sleeping_dog"] < -0.1, mean_by_segment


# ----------------------------------------------------------------- Qini
def test_true_uplift_ordering_beats_random_and_random_is_near_zero():
    # Over the 100-seed sweep: true-uplift Qini 0.024 to 0.032; every single
    # random ordering within +-0.0066, the mean of 20 within +-0.0011.
    q_true, q_random = strong_campaign_qini()
    assert q_true > 0.015
    assert (q_random < q_true / 2).all()
    assert abs(q_random.mean()) < 0.003


# ---------------------------------------------------------- end to end
def test_uplift_evaluation_end_to_end(pipeline_run, raw_csv, monkeypatch):
    """evaluate.main() on the synthetic pipeline's saved models, writing only to
    the pipeline's temporary project folder."""
    import uplift.evaluate as uplift_evaluate

    root = pipeline_run.root
    for name, path in {"ROOT": root, "FIGURES_DIR": root / "reports" / "figures",
                       "RESULTS_DIR": root / "results", "MODELS_DIR": root / "models",
                       "METRICS_PATH": root / "results" / "metrics.csv"}.items():
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(uplift_evaluate, "load_train_test",
                        lambda: split(load_and_clean(raw_csv)))
    uplift_evaluate.main()

    assert (root / "reports" / "figures" / "12_qini_curve.png").exists()
    table = pd.read_csv(root / "results" / "uplift_targeting.csv")
    assert table["scenario"].unique().tolist() == list(config.UPLIFT_SCENARIOS)
    assert (table["contacted"] == round(len(pipeline_run.X_test)
                                        * config.UPLIFT_TARGET_FRACTION)).all()
    for _, scenario in table.groupby("scenario"):
        assert scenario["rule"].tolist() == list(uplift_evaluate.RULES)
        by_rule = scenario.set_index("rule")
        # The true-uplift ordering is the best any rule can do at this budget.
        assert by_rule.loc["true_uplift", "share_of_best"] == pytest.approx(1.0)
        assert (by_rule["incremental_retained"]
                <= by_rule.loc["true_uplift", "incremental_retained"] + 1e-9).all()
    campaign = pd.read_csv(root / "results" / "uplift_campaign_test.csv")
    assert len(campaign) == len(pipeline_run.X_test)
    assert {"treatment", "true_uplift", "predicted_uplift"} <= set(campaign.columns)


def test_sensitivity_scenario_only_changes_lost_causes_and_persuadables():
    main = config.UPLIFT_SCENARIOS["main"]
    alt = config.UPLIFT_SCENARIOS["sensitivity_high_risk_responds"]
    assert main is config.UPLIFT_SEGMENT_EFFECTS
    assert {s for s in main if main[s] != alt[s]} == {"lost_cause", "persuadable"}
    assert alt["lost_cause"] > alt["persuadable"]
