
# Customer Churn Prediction

## Project Overview
This project uses traditional Machine Learning to predict whether a telecom customer is likely to leave the company (Churn).

## Dataset
- Dataset: Cell2Cell Telecom Churn
- Number of records: 51,047
- Target variable: Churn

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

## Final Model
Decision Tree was selected because it achieved the highest Recall among the tested models.

## Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Deployment
- FastAPI provides the prediction API.
- Streamlit provides the user interface.

## Project Files
- `Customer_Churn_Prediction_NTI_Project.ipynb`: project workflow.
- `api.py`: FastAPI application.
- `app.py`: Streamlit application.
- `feature_config.json`: input feature configuration.
- `customer_churn_model.pkl`: trained Decision Tree model.
- `customer_churn_preprocessor.pkl`: fitted preprocessing pipeline.
- `requirements.txt`: required Python libraries.

## How to Run

Install dependencies:

    pip install -r requirements.txt

Start the API in one terminal:

    uvicorn api:app --host 127.0.0.1 --port 8000

Start the Streamlit app in a second terminal:

    streamlit run app.py

Keep both processes running while using the application.
