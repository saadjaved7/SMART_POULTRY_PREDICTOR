from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from datetime import datetime
import os

# =====================================================
# FASTAPI APP SETUP
# =====================================================
app = FastAPI(title="Smart Poultry Prediction API")

# =====================================================
# API KEY
# =====================================================
API_KEY = "mysecretkey123"

# =====================================================
# LOAD PKL MODELS
# =====================================================
models = {}
model_file = "lahore_rawalpindi_models.pkl"

if not os.path.exists(model_file):
    raise Exception(f"❌ Model file '{model_file}' not found!")

try:
    with open(model_file, 'rb') as f:
        models = pickle.load(f)
    print(f"✅ Loaded models: {list(models.keys())}")
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# HELPER: CREATE DUMMY FEATURES
# =====================================================
def create_features(date_str: str):
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    day_of_week = target_date.weekday()
    month = target_date.month
    day = target_date.day
    
    features = [
        day_of_week,
        month,
        day,
        350.0,  # lag_1 (dummy)
        348.0,  # lag_7 (dummy)
        349.5,  # ma_7 (dummy)
        349.2,  # ema_7 (dummy)
        0.5,    # momentum (dummy)
        1.2,    # volatility (dummy)
        1.0,    # trend (dummy)
    ]
    
    return np.array([features])

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    return {
        "message": "🐔 Smart Poultry Prediction API",
        "available_cities": list(models.keys()),
        "example_usage": "https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-02-20&api_key=mysecretkey123",
        "note": "Include api_key in URL"
    }

@app.get("/predict_date")
def predict_date(city: str, date: str, api_key: str):
    """
    Predict poultry prices for a given city and date.
    
    Example: /predict_date?city=Lahore&date=2025-02-20&api_key=mysecretkey123
    """
    
    # Check API key
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. Please provide valid api_key parameter."
        )
    
    # Find city (case-insensitive)
    city_key = None
    for key in models.keys():
        if key.lower() == city.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. Available: {list(models.keys())}"
        )
    
    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD (e.g., 2025-02-20)"
        )
    
    # Get city models
    city_models = models[city_key]
    open_model = city_models['Open']
    close_model = city_models['Close']
    
    # Create features
    X = create_features(date)
    
    # Make predictions
    try:
        predicted_open = float(open_model.predict(X)[0])
        predicted_close = float(close_model.predict(X)[0])
        change = predicted_close - predicted_open
        
        return {
            "city": city_key,
            "date": date,
            "predictions": {
                "open": round(predicted_open, 2),
                "close": round(predicted_close, 2),
                "expected_change": round(change, 2)
            },
            "currency": "PKR"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


## 🌐 **Now Use These URLs Directly in Browser:**

# ### **Test Lahore:**
# ```
# https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-02-20&api_key=mysecretkey123
# ```

# ### **Test Rawalpindi:**
# ```
# https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Rawalpindi&date=2025-11-20&api_key=mysecretkey123

