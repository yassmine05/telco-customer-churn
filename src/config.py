"""Central configuration: paths, random seed, feature lists and cost assumptions."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
MODELS_DIR = ROOT / "models"

# --- Reproducibility ---------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# Nested CV: outer folds estimate performance; inner folds (CV_FOLDS) tune.
# The outer splitter uses SEED + 1 so its folds differ from the inner ones.
OUTER_CV_FOLDS = 5

# --- Columns -----------------------------------------------------------------
TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is already 0/1 in the raw file, so it is passed through as-is.
PASSTHROUGH_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# --- Business cost assumptions (used to pick the decision threshold) ---------
# A missed churner (false negative) is assumed to cost 5x as much as a
# retention offer sent to a customer who would have stayed (false positive).
# This is an ASSUMPTION, not something the dataset tells us. Change it here
# and re-run to see how the threshold moves.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# --- XGBoost search space (tuned with CV on the training set only) -----------
XGB_PARAM_GRID = {
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__n_estimators": [100, 200, 300, 500, 800],
}

# --- Uplift modelling on a SIMULATED retention campaign ----------------------
# The Telco data has no treatment column: nobody in it was (or was not) sent a
# retention offer. src/uplift/simulate.py therefore invents a randomised
# campaign. Everything below is an ASSUMPTION chosen to exercise the method;
# none of it is a fact about real customers.
UPLIFT_SEED = 7
UPLIFT_TREATMENT_SHARE = 0.5      # share of customers randomly sent the offer
UPLIFT_TARGET_FRACTION = 0.20     # campaign budget: contact the top 20%

# Segments are assigned in this order; the first matching rule wins.
#   sure_thing    two-year contract: low churn with or without the offer.
#   sleeping_dog  month-to-month and tenure >= 48 months: loyal out of habit;
#                 the offer reminds them they are free to leave.
#   lost_cause    month-to-month, fiber optic and electronic check: high churn
#                 that the offer does not change.
#   persuadable   month-to-month, tenure <= 24 months, no tech support: the
#                 offer cuts churn a lot.
#   other         everyone else: a small benefit.
UPLIFT_SLEEPING_DOG_MIN_TENURE = 48
UPLIFT_PERSUADABLE_MAX_TENURE = 24
# True treatment effect per segment, in percentage points of churn probability:
#   P(churn | no offer) - P(churn | offer). Positive means the offer helps;
#   negative means it backfires. The treated probability is clipped to [0, 1],
#   so a customer's actual effect can be smaller than the segment's value.
UPLIFT_SEGMENT_EFFECTS = {
    "sure_thing": 0.0,
    "sleeping_dog": -0.05,
    "lost_cause": 0.0,
    "persuadable": 0.20,
    "other": 0.03,
}
# SENSITIVITY SCENARIO (not the main result): the offer works best on the
# riskiest customers. Same segments and rules; only lost causes (+0.20) and
# persuadables (+0.05) change, so risk and uplift line up. It checks how much
# the targeting comparison depends on the effect sizes assumed above.
UPLIFT_SENSITIVITY_EFFECTS = {
    **UPLIFT_SEGMENT_EFFECTS,
    "lost_cause": 0.20,
    "persuadable": 0.05,
}
UPLIFT_SCENARIOS = {
    "main": UPLIFT_SEGMENT_EFFECTS,
    "sensitivity_high_risk_responds": UPLIFT_SENSITIVITY_EFFECTS,
}
