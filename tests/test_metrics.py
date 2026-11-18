"""Tests for src/metrics.py on small hand-built examples with known answers."""
import numpy as np
import pandas as pd
import pytest

import config
import metrics

# 10 customers, 4 churners. At threshold 0.5:
#   churners scored 0.9, 0.8, 0.7 -> caught (TP = 3); 0.2 -> missed (FN = 1)
#   stayers scored 0.6, 0.5 -> flagged (FP = 2); 0.4, 0.3, 0.1, 0.0 -> fine (TN = 4)
Y_HAND = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
P_HAND = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.5, 0.4, 0.3, 0.1, 0.0])


# --------------------------------------------------------------- cost
def test_expected_cost_matches_hand_computed_confusion_matrix():
    # 1 missed churner x 5 + 2 wasted offers x 1 = 7
    assert metrics.expected_cost(Y_HAND, P_HAND, 0.5) == pytest.approx(7.0)
    assert (config.COST_FALSE_NEGATIVE * 1 + config.COST_FALSE_POSITIVE * 2) == 7.0


def test_expected_cost_extremes():
    # Flag nobody: every churner missed. Flag everybody: every stayer gets an offer.
    assert metrics.expected_cost(Y_HAND, P_HAND, 1.01) == 4 * 5.0
    assert metrics.expected_cost(Y_HAND, P_HAND, 0.0) == 6 * 1.0


def test_expected_cost_accepts_pandas_series():
    y = pd.Series(Y_HAND, index=range(100, 110))   # non-default index, like a split
    assert metrics.expected_cost(y, P_HAND, 0.5) == pytest.approx(7.0)


def test_classification_metrics_on_hand_example():
    m = metrics.classification_metrics(Y_HAND, P_HAND, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (4, 2, 1, 3)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    # ROC-AUC = share of (churner, stayer) pairs ranked correctly. Churners at
    # 0.9/0.8/0.7 beat all 6 stayers (18 pairs); 0.2 beats only 0.1 and 0.0 (2).
    assert m["roc_auc"] == pytest.approx(20 / 24)


def test_score_equal_to_threshold_counts_as_flagged():
    m = metrics.classification_metrics(Y_HAND, P_HAND, 0.6)
    assert m["fp"] == 1   # the stayer scored exactly 0.6 is flagged


# ---------------------------------------------------------- threshold
# Churners scored 0.9 and 0.35; stayers 0.6, 0.5 and 0.1. With a 5:1 cost:
#   t <= 0.10        -> FP = 3          cost 3
#   0.10 < t <= 0.35 -> FP = 2          cost 2   <- minimum
#   0.35 < t <= 0.50 -> FN = 1, FP = 2  cost 7
#   0.50 < t <= 0.60 -> FN = 1, FP = 1  cost 6
#   0.60 < t <= 0.90 -> FN = 1          cost 5
Y_TOY = np.array([1, 1, 0, 0, 0])
P_TOY = np.array([0.9, 0.35, 0.6, 0.5, 0.1])


def test_best_threshold_on_toy_example_with_5_to_1_cost():
    # Several thresholds tie at the minimum; the lowest one (highest recall) wins.
    assert metrics.best_cost_threshold(Y_TOY, P_TOY) == pytest.approx(0.11)


def test_best_threshold_is_the_cost_minimum_of_the_grid():
    grid, costs = metrics.cost_curve(Y_TOY, P_TOY)
    chosen = metrics.best_cost_threshold(Y_TOY, P_TOY)
    assert metrics.expected_cost(Y_TOY, P_TOY, chosen) == costs.min() == 2.0
    brute_force = min(grid, key=lambda t: (metrics.expected_cost(Y_TOY, P_TOY, t), t))
    assert chosen == pytest.approx(brute_force)


def test_best_threshold_follows_the_cost_ratio_in_config(monkeypatch):
    # With equal costs, missing the 0.35 churner is cheaper than two wasted
    # offers, so the best threshold moves up to 0.61 (cost 1).
    monkeypatch.setattr(config, "COST_FALSE_NEGATIVE", 1.0)
    assert metrics.best_cost_threshold(Y_TOY, P_TOY) == pytest.approx(0.61)


def test_cost_curve_default_grid():
    grid, costs = metrics.cost_curve(Y_TOY, P_TOY)
    assert len(grid) == len(costs) == 91
    assert grid[0] == pytest.approx(0.05) and grid[-1] == pytest.approx(0.95)


# ------------------------------------------------------- capture / lift
# Ranked riskiest first, the 10 customers' true labels are:
#   rank:   1  2  3  4  5  6  7  8  9  10
#   churn:  1  0  1  1  0  0  1  0  0  0     (4 churners)
RANKED_LABELS = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
RANKED_SCORES = np.linspace(0.95, 0.05, 10)


@pytest.fixture
def shuffled():
    """The same customers in scrambled order, so the function must sort them."""
    order = np.random.default_rng(0).permutation(10)
    return RANKED_LABELS[order], RANKED_SCORES[order]


@pytest.mark.parametrize("fraction, contacted, found, capture, precision, lift", [
    (0.10, 1, 1, 0.25, 1.0, 2.5),
    (0.20, 2, 1, 0.25, 0.5, 1.25),
    (0.40, 4, 3, 0.75, 0.75, 1.875),
    (1.00, 10, 4, 1.00, 0.4, 1.0),
])
def test_capture_and_lift_on_hand_ranking(shuffled, fraction, contacted, found,
                                          capture, precision, lift):
    y, p = shuffled
    out = metrics.capture_at_top(y, p, fraction)
    assert out["contacted"] == contacted
    assert out["churners_found"] == found
    assert out["capture"] == pytest.approx(capture)
    assert out["precision_in_group"] == pytest.approx(precision)
    assert out["lift"] == pytest.approx(lift)


def test_random_scores_give_lift_near_one():
    rng = np.random.default_rng(1)
    y = rng.random(20_000) < 0.27
    out = metrics.capture_at_top(y, rng.random(20_000), 0.20)
    assert out["lift"] == pytest.approx(1.0, abs=0.06)


# -------------------------------------------------------------- logging
def test_log_metrics_appends_with_a_single_header(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "results" / "metrics.csv")
    metrics.log_metrics([{"model": "a", "roc_auc": 0.81234567}])
    metrics.log_metrics([{"model": "b", "roc_auc": 0.7}])
    df = pd.read_csv(tmp_path / "results" / "metrics.csv")
    assert list(df["model"]) == ["a", "b"]
    assert list(df.columns[:2]) == ["run_time", "seed"]
    assert (df["seed"] == config.SEED).all()
    assert df.loc[0, "roc_auc"] == 0.8123     # written with 4 decimals


def test_log_metrics_adds_a_new_column_without_misaligning_old_rows(tmp_path, monkeypatch):
    # Older runs did not log a Brier score; a run that does must widen the
    # file rather than shift its values under the wrong headers.
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "results" / "metrics.csv")
    metrics.log_metrics([{"model": "old", "roc_auc": 0.8, "params": "{}"}])
    metrics.log_metrics([{"model": "new", "roc_auc": 0.7, "brier": 0.15, "params": "{}"}])
    metrics.log_metrics([{"model": "newer", "roc_auc": 0.6, "brier": 0.12, "params": "{}"}])
    df = pd.read_csv(tmp_path / "results" / "metrics.csv")
    assert list(df.columns) == ["run_time", "seed", "model", "roc_auc", "params", "brier"]
    assert list(df["model"]) == ["old", "new", "newer"]
    assert list(df["roc_auc"]) == [0.8, 0.7, 0.6]
    assert df["brier"].isna().tolist() == [True, False, False]
    assert list(df["brier"].iloc[1:]) == [0.15, 0.12]


# --------------------------------------------------------------- Qini
# Six customers, already in score order; T = sent the offer, C = control.
#   k  arm  churn | N_T C_T N_C C_C | Q = C_C * N_T / N_C - C_T
#   1  T    0     |  1   0   0   0  | 0 (no control yet)
#   2  C    1     |  1   0   1   1  | 1 * 1/1 - 0 = 1
#   3  T    0     |  2   0   1   1  | 1 * 2/1 - 0 = 2
#   4  C    1     |  2   0   2   2  | 2 * 2/2 - 0 = 2
#   5  T    1     |  3   1   2   2  | 2 * 3/2 - 1 = 2
#   6  C    0     |  3   1   3   2  | 2 * 3/3 - 1 = 1
Q_CHURN = np.array([0, 1, 0, 1, 1, 0])
Q_TREAT = np.array([1, 0, 1, 0, 1, 0])
Q_SCORE = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def test_qini_curve_on_hand_example():
    fraction, q = metrics.qini_curve(Q_CHURN, Q_TREAT, Q_SCORE)
    np.testing.assert_allclose(fraction, np.arange(7) / 6)
    np.testing.assert_allclose(q, [0, 0, 1, 2, 2, 2, 1])


def test_qini_curve_sorts_by_score_not_input_order():
    perm = np.array([3, 0, 5, 1, 4, 2])
    _, q = metrics.qini_curve(Q_CHURN[perm], Q_TREAT[perm], Q_SCORE[perm])
    np.testing.assert_allclose(q, [0, 0, 1, 2, 2, 2, 1])


def test_qini_coefficient_on_hand_example():
    # Trapezoid area under Q / n with dx = 1/6:
    #   (1/6) * (0/2 + 0 + 1 + 2 + 2 + 2 + 1/2) / 6 = 7.5 / 36
    # Random line from (0, 0) to (1, 1/6): area 1/12 = 3 / 36.
    # Coefficient = 4.5 / 36 = 0.125.
    assert metrics.qini_coefficient(Q_CHURN, Q_TREAT, Q_SCORE) == pytest.approx(0.125)


def test_qini_coefficient_is_negative_for_the_reversed_ordering():
    # Reversed scores visit the rows bottom-up: (C,0), (T,1), (C,1), (T,0),
    # (C,1), (T,0), as (arm, churn). Q after each row:
    #   1: 0 * 0/1 - 0 = 0         4: 1 * 2/2 - 1 = 0
    #   2: 0 * 1/1 - 1 = -1        5: 2 * 2/3 - 1 = 1/3
    #   3: 1 * 1/2 - 1 = -1/2      6: 2 * 3/3 - 1 = 1
    # Area under Q / n: (1/6) * (0 - 1 - 1/2 + 0 + 1/3 + 1/2) / 6 = (-2/3) / 36.
    # Minus the random line's 3 / 36: -(11/3) / 36 = -11/108.
    _, q = metrics.qini_curve(Q_CHURN, Q_TREAT, -Q_SCORE)
    np.testing.assert_allclose(q, [0, 0, -1, -0.5, 0, 1 / 3, 1])
    assert metrics.qini_coefficient(Q_CHURN, Q_TREAT, -Q_SCORE) == pytest.approx(-11 / 108)
