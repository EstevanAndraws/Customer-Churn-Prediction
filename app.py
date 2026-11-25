
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

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
        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Result")

        if result["churn_status"] == "Yes":
            st.error("⚠️ Customer is likely to Churn!")

        else:
            st.success("✅ Customer is likely to Stay!")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
