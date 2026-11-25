# Customer Churn Prediction

## 1. Project Overview
This project uses Traditional Machine Learning techniques to predict whether a telecom customer is likely to leave the company (Customer Churn).

The goal is to identify customers at risk of leaving and help telecom companies improve customer retention.

## 2. Dataset
- **Dataset:** Cell2Cell Telecom Churn
- **Number of Records:** 51,047
- **Target Variable:** Churn
- **Features:** Customer demographics, account information, service usage, and customer behavior.

## 3. Data Preprocessing & Feature Engineering
- Handled missing values.
- Identified and treated invalid negative values in selected features.
- Encoded categorical variables.
- Scaled numerical features.
- Created additional features to improve model performance.
- Applied a stratified train-test split.

## 4. Machine Learning Models
The following Traditional Machine Learning models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

## 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5785 | 0.3567 | 0.5761 | 0.4406 | 0.6103 |
| Decision Tree | 0.5657 | 0.3577 | 0.6377 | 0.4583 | 0.6163 |
| Random Forest | 0.6392 | 0.4037 | 0.5289 | 0.4579 | 0.6558 |
| Tuned Random Forest | 0.6984 | 0.4645 | 0.3049 | 0.3682 | 0.6622 |

## 6. Final Model Selection

The Decision Tree model was selected as the final model because it achieved the highest Recall among the evaluated models.

Recall is particularly important in customer churn prediction because identifying customers who are likely to leave helps businesses take retention actions.

## 7. Deployment
The project includes a simple deployment solution:

- **FastAPI:** Provides the prediction API.
- **Streamlit:** Provides an interactive user interface.
- **Joblib:** Used to load the trained model and preprocessing pipeline.

## 8. Project Files

- `Customer_Churn_Prediction_NTI_Project.ipynb`: Complete project workflow.
- `api.py`: FastAPI application.
- `app.py`: Streamlit application.
- `customer_churn_model.pkl`: Trained Decision Tree model.
- `customer_churn_preprocessor.pkl`: Fitted preprocessing pipeline.
- `feature_config.json`: Input feature configuration.
- `requirements.txt`: Required Python libraries.

## 9. How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the FastAPI Server

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Start the Streamlit Application

Open a second terminal and run:

```bash
streamlit run app.py
```

Keep both the API server and Streamlit application running while using the project.

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Joblib
- Jupyter Notebook
