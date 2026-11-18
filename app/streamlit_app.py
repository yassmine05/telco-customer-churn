"""Minimal Streamlit app: enter one customer's details, get a calibrated churn
probability and a SHAP explanation of the underlying model score.

Run:  streamlit run app/streamlit_app.py     (after python src/run_all.py)
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from explain import explain  # noqa: E402
from schema import INFERENCE_SCHEMA, DataValidationError, validate  # noqa: E402

st.set_page_config(page_title="Telco churn risk", layout="wide")


MODEL_FILES = ["xgboost.joblib", "xgboost_calibrated.joblib", "metadata.json"]


@st.cache_resource
def load_models():
    # The raw booster is kept for SHAP; the calibrated model gives the probability.
    pipeline = joblib.load(config.MODELS_DIR / "xgboost.joblib")
    calibrated = joblib.load(config.MODELS_DIR / "xgboost_calibrated.joblib")
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    return pipeline, calibrated, meta


if not all((config.MODELS_DIR / f).exists() for f in MODEL_FILES):
    st.error("No trained model found. Run `python src/run_all.py` first.")
    st.stop()

pipeline, calibrated, meta = load_models()
threshold = meta["thresholds"]["xgboost_calibrated"]

st.title("Telco churn risk")
st.caption("Enter a customer's details to get the model's churn probability and "
           "the factors behind it.")

# ------------------------------------------------------------------ inputs
YES_NO = ["No", "Yes"]
with st.sidebar:
    st.header("Customer details")
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0, step=1.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", YES_NO, index=1)

    st.subheader("Services")
    phone = st.selectbox("Phone service", YES_NO, index=1)
    multiple = (st.selectbox("Multiple lines", YES_NO) if phone == "Yes"
                else "No phone service")
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addons = {}
    for label, column in [("Online security", "OnlineSecurity"),
                          ("Online backup", "OnlineBackup"),
                          ("Device protection", "DeviceProtection"),
                          ("Tech support", "TechSupport"),
                          ("Streaming TV", "StreamingTV"),
                          ("Streaming movies", "StreamingMovies")]:
        addons[column] = (st.selectbox(label, YES_NO) if internet != "No"
                          else "No internet service")

    st.subheader("Profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", YES_NO)
    partner = st.selectbox("Partner", YES_NO)
    dependents = st.selectbox("Dependents", YES_NO)

customer = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multiple, "InternetService": internet, **addons,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    # Approximation: the form does not ask for billing history.
    "TotalCharges": monthly * tenure,
}])

# ------------------------------------------------------------------ output
# Validate before scoring: an unknown level in a Yes/No column would otherwise
# be scored silently as the dropped level (see src/schema.py).
try:
    validate(customer, INFERENCE_SCHEMA)
except DataValidationError as error:
    st.error("This customer cannot be scored because the input failed validation:\n\n"
             + str(error))
    st.stop()

probability = float(calibrated.predict_proba(customer)[0, 1])
raw_score = float(pipeline.predict_proba(customer)[0, 1])
flagged = probability >= threshold

left, right = st.columns([1, 2])
with left:
    st.metric("Churn probability", f"{probability:.0%}")
    if flagged:
        st.warning(f"Above the action threshold ({threshold:.2f}): "
                   "candidate for a retention offer.")
    else:
        st.success(f"Below the action threshold ({threshold:.2f}): no offer.")
    st.caption(
        f"Calibrated ({meta['calibration']['method']}) on the training set: among "
        "customers given this probability, roughly that share churned. The "
        f"threshold is where a {config.COST_FALSE_NEGATIVE:g}:"
        f"{config.COST_FALSE_POSITIVE:g} cost of a missed churner vs a wasted "
        "offer is lowest.")

with right:
    st.subheader("What drives the model's raw score")
    explanation = explain(pipeline, customer)
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        f"Red bars push the score up, blue bars push it down (units: log-odds). "
        f"This explains the uncalibrated model's raw score ({raw_score:.0%}), "
        "not the calibrated probability; the two rank customers almost the same "
        "way. SHAP shows what the model relies on, not what causes churn.")
