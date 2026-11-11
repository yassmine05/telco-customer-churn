"""Minimal Streamlit app: enter one customer's details, get a churn risk score
and a SHAP explanation.

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

st.set_page_config(page_title="Telco churn risk", layout="wide")


@st.cache_resource
def load_model():
    pipeline = joblib.load(config.MODELS_DIR / "xgboost.joblib")
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    return pipeline, meta


if not (config.MODELS_DIR / "xgboost.joblib").exists():
    st.error("No trained model found. Run `python src/run_all.py` first.")
    st.stop()

pipeline, meta = load_model()
threshold = meta["thresholds"]["xgboost"]

st.title("Telco churn risk")
st.caption("Enter a customer's details to get the model's churn risk score and "
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
score = float(pipeline.predict_proba(customer)[0, 1])
flagged = score >= threshold

left, right = st.columns([1, 2])
with left:
    st.metric("Churn risk score", f"{score:.0%}")
    if flagged:
        st.warning(f"Above the action threshold ({threshold:.2f}): "
                   "candidate for a retention offer.")
    else:
        st.success(f"Below the action threshold ({threshold:.2f}): no offer.")
    st.caption(
        "The score ranks customers by risk. It is not a calibrated probability: "
        "the model up-weights churners during training, so scores run higher "
        "than true churn rates.")

with right:
    st.subheader("What drives this score")
    explanation = explain(pipeline, customer)
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Red bars push the score up, blue bars push it down (units: log-odds). "
        "SHAP shows what the model relies on, not what causes churn.")
