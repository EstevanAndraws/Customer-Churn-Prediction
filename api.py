
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Machine Learning",
    version="1.0.0"
)

final_model = joblib.load("customer_churn_model.pkl")
preprocessor = joblib.load("customer_churn_preprocessor.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running successfully!"}

@app.post("/predict")
def predict_churn(customer_data: Dict[str, Any]):
    try:
        input_data = pd.DataFrame([customer_data])

        input_data["AverageMonthlyUsage"] = (
            input_data["MonthlyMinutes"] /
            input_data["MonthsInService"].replace(0, float("nan"))
        )
        input_data["OverageRatio"] = (
            input_data["OverageMinutes"] /
            (input_data["MonthlyMinutes"] + 1)
        )
        input_data["RetentionEngagement"] = (
            (input_data["RetentionCalls"] > 0) |
            (input_data["RetentionOffersAccepted"] > 0)
        ).astype(int)

        processed_data = preprocessor.transform(input_data)
        prediction = final_model.predict(processed_data)[0]

        return {
            "prediction": int(prediction),
            "churn_status": "Yes" if prediction == 1 else "No"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
