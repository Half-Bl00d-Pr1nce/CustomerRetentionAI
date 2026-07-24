import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from utils import create_engineered_features

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------
# Load Model
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

model = joblib.load(MODEL_PATH)

# ------------------------------------
# Header
# ------------------------------------
st.title("📉 Customer Churn Prediction System")

st.markdown(
"""
Predict whether a telecom customer is likely to churn using the trained
**Logistic Regression + SMOTE** model.
"""
)

# ------------------------------------
# Sidebar
# ------------------------------------
st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Model
- Logistic Regression
- SMOTE

### Dataset
IBM Telco Customer Churn

### Developed Using
- Streamlit
- Scikit-Learn
- Pandas
""")

# ------------------------------------
# Customer Input
# ------------------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes","No","No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )

with col2:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes","No","No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=850.0
    )

# ------------------------------------
# Predict Button
# ------------------------------------
if st.button("Predict Churn"):

    input_df = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":[OnlineSecurity],
        "OnlineBackup":[OnlineBackup],
        "DeviceProtection":[DeviceProtection],
        "TechSupport":[TechSupport],
        "StreamingTV":[StreamingTV],
        "StreamingMovies":[StreamingMovies],
        "Contract":[Contract],
        "PaperlessBilling":[PaperlessBilling],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges],
        "TotalCharges":[TotalCharges]

    })

    # --------------------------------
    # Feature Engineering
    # --------------------------------

    input_df = create_engineered_features(input_df)

    st.subheader("Customer Profile")

    st.dataframe(input_df)

    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.header("Prediction Results")

    c1,c2 = st.columns(2)

    with c1:

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with c2:

        if probability >= 0.75:

            risk = "🔴 High"

        elif probability >= 0.50:

            risk = "🟡 Medium"

        else:

            risk = "🟢 Low"

        st.metric(
            "Risk Level",
            risk
        )

    # --------------------------------
    # Prediction Text
    # --------------------------------

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn.")

    else:

        st.success("✅ Customer is likely to stay.")

    # --------------------------------
    # Recommendation
    # --------------------------------

    st.subheader("Business Recommendation")

    if probability >= 0.75:

        st.info("""
**Recommended Action**

• Contact customer immediately

• Offer personalized discount

• Assign customer success representative

• Review available contract upgrades
""")

    elif probability >= 0.50:

        st.warning("""
**Recommended Action**

• Offer promotional plans

• Increase customer engagement

• Follow up through customer support
""")

    else:

        st.success("""
**Recommended Action**

Customer appears satisfied.

Continue regular engagement and monitor future activity.
""")