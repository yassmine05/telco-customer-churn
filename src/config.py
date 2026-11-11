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
