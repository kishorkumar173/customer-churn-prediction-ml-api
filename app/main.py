from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

from src.feature_engineering import feature_engineering

# =========================
# LOAD MODEL + COLUMNS
# =========================
model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# INPUT SCHEMA
# =========================
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running 🚀"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: Customer):

    try:
        # Convert input → DataFrame
        df = pd.DataFrame([data.dict()])

        # Apply feature engineering
        df = feature_engineering(df)

        # Align columns (VERY IMPORTANT)
        df = df.reindex(columns=columns, fill_value=0)

        # Predict
        prob = model.predict_proba(df)[0][1]
        prediction = int(prob > 0.5)

        return {
            "churn_probability": round(prob, 3),
            "churn_prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}