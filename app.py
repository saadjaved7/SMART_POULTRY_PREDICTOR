from fastapi import FastAPI, HTTPException, Query
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# =====================================================
#                 FASTAPI APP SETUP
# =====================================================
app = FastAPI(title="Smart Poultry Prediction API")

# =====================================================
#                LOAD PICKLED MODELS
# =====================================================
models = {}
model_file = "lahore_rawalpindi_models.pkl"  # your .pkl file
if not os.path.exists(model_file):
    raise Exception(f"❌ Model file '{model_file}' not found in repo!")

with open(model_file, "rb") as f:
    models = pickle.load(f)

print(f"✅ Loaded models: {list(models.keys())}")

# =====================================================
#                MOCK LATEST HISTORICAL DATA
# =====================================================
latest_data = {
    "Rawalpindi": {"Open": 350.00, "Close": 350.00, "Date": "2025-11-11"},
    "Lahore": {"Open": 355.00, "Close": 357.00, "Date": "2025-11-11"},
}

# =====================================================
#                        ROUTES
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Poultry Prediction API!",
        "available_cities": list(models.keys())
    }

@app.get("/predict_date")
def predict_date(
    city: str = Query(..., description="City name, e.g., Lahore or Rawalpindi"),
    date: str = Query(..., description="Date in YYYY-MM-DD format")
):
    """
    Predict Open and Close prices for a city on a given date.
    Example:
    GET /predict_date?city=Lahore&date=2025-02-20
    """
    city = city.title()  # normalize input
    if city not in models:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. Available: {list(models.keys())}"
        )

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format! Use YYYY-MM-DD.")

    city_models = models[city]  # expects dict: {"Open": model, "Close": model, "metrics": {...}}

    # Mock features (replace with your real feature engineering)
    X = np.array([[0, 0, 0, 0]])  # dummy features

    open_model = city_models.get("Open")
    close_model = city_models.get("Close")
    metrics = city_models.get("metrics", {})

    if open_model is None or close_model is None:
        raise HTTPException(status_code=500, detail="City models incomplete (Open/Close missing).")

    # Predictions
    predicted_open = float(open_model.predict(X)[0])
    predicted_close = float(close_model.predict(X)[0])
    expected_change = predicted_close - predicted_open

    latest = latest_data.get(city, {"Open": None, "Close": None, "Date": None})

    return {
        "city": city,
        "date": str(target_date),
        "prediction": {
            "Open": round(predicted_open, 2),
            "Close": round(predicted_close, 2),
            "Expected_Change": round(expected_change, 2)
        },
        "latest_data": latest,
        "metrics": metrics,
        "api_key_used": "secretapikey123"  # automatically included
    }
