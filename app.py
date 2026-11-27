import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load("customer_churn_model.pkl")
    preprocessor = joblib.load("customer_churn_preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# Load feature configuration
with open("feature_config.json", "r") as f:
    feature_config = json.load(f)

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to leave the company.")

st.divider()

customer_data = {}

with st.form("customer_form"):

    st.subheader("Customer Information")

    columns = st.columns(2)

    for i, (feature, config) in enumerate(feature_config.items()):

        with columns[i % 2]:

            if config["type"] == "numeric":
                customer_data[feature] = st.number_input(
                    feature,
                    value=float(config["default"])
                )

            else:
                customer_data[feature] = st.selectbox(
                    feature,
                    options=config["options"],
                    index=config["options"].index(config["default"])
                )

    submitted = st.form_submit_button("🔍 Predict Churn")

if submitted:

    try:
        # Convert customer input into DataFrame
        input_df = pd.DataFrame([customer_data])

        # Feature Engineering
        input_df["AverageMonthlyUsage"] = (
            input_df["MonthlyMinutes"] /
            input_df["MonthsInService"].replace(0, np.nan)
        )

        input_df["OverageRatio"] = (
            input_df["OverageMinutes"] /
            (input_df["MonthlyMinutes"] + 1)
        )

        input_df["RetentionEngagement"] = (
            (input_df["RetentionCalls"] > 0) |
            (input_df["RetentionOffersAccepted"] > 0)
        ).astype(int)

        # Apply preprocessing
        processed_data = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to Churn!")

        else:
            st.success("✅ Customer is likely to Stay!")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
