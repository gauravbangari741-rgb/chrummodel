from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please train the model first using model/train_model.py")

# Pydantic model for input validation
class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int  # 0 or 1
    IsActiveMember: int  # 0 or 1
    EstimatedSalary: float
    Geography: str  # 'France', 'Germany', 'Spain'
    Gender: str  # 'Male', 'Female'

def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """Convert input data to DataFrame and align columns with model features."""
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Encode categorical variables (same as training)
    df = pd.get_dummies(df, drop_first=True)
    
    # Align columns with model's feature names
    feature_names = model.feature_names_in_
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    
    return df

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "API is running", "message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict_churn(customer: CustomerData) -> Dict[str, Any]:
    """Predict customer churn based on input data."""
    try:
        # Preprocess input
        input_df = preprocess_input(customer)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of churn (class 1)
        
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the app with: uvicorn main:app --reload