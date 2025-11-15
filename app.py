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
    
    # Debug: Print structure
    for city in models.keys():
        print(f"   {city}:")
        for price_type in models[city].keys():
            model_data = models[city][price_type]
            print(f"      {price_type} ({model_data['type']}): R²={model_data['r2']:.4f}, MAE={model_data['mae']:.2f}")
            
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# HELPER: CREATE FEATURES
# =====================================================
def create_features(date_str: str, feature_names: list):
    """
    Creates features matching the training data.
    Uses dummy values for lag/MA/EMA features.
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Time features
    day_of_week = target_date.weekday()
    month = target_date.month
    day = target_date.day
    
    # Create feature dict with dummy values
    features_dict = {
        'day_of_week': day_of_week,
        'month': month,
        'day': day,
        'lag_1': 350.0,
        'lag_2': 349.0,
        'lag_3': 348.5,
        'lag_7': 348.0,
        'lag_14': 347.5,
        'ma_7': 349.5,
        'ma_14': 349.2,
        'ema_7': 349.2,
        'ema_14': 349.0,
        'momentum_7': 0.5,
        'momentum_14': 0.4,
        'volatility_7': 1.2,
        'volatility_14': 1.3,
        'trend': 1.0,
        'cross_city_lag_1': 350.5,
        'cross_city_lag_7': 349.0,
    }
    
    # Create array in the correct order based on feature_names
    feature_array = []
    for feature_name in feature_names:
        if feature_name in features_dict:
            feature_array.append(features_dict[feature_name])
        else:
            # Default value for unknown features
            feature_array.append(0.0)
    
    return np.array([feature_array])

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    model_info = {}
    for city in models.keys():
        model_info[city] = {
            "Open": {
                "type": models[city]['Open']['type'],
                "r2": round(models[city]['Open']['r2'], 4),
                "mae": round(models[city]['Open']['mae'], 2)
            },
            "Close": {
                "type": models[city]['Close']['type'],
                "r2": round(models[city]['Close']['r2'], 4),
                "mae": round(models[city]['Close']['mae'], 2)
            }
        }
    
    return {
        "message": "🐔 Smart Poultry Prediction API",
        "available_cities": list(models.keys()),
        "model_info": model_info,
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
    try:
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']
        
        open_model = open_data['model']
        close_model = close_data['model']
        open_scaler = open_data['scaler']
        close_scaler = close_data['scaler']
        open_features = open_data['features']
        close_features = close_data['features']
        
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing model structure: {str(e)}"
        )
    
    # Create features
    try:
        X_open = create_features(date, open_features)
        X_close = create_features(date, close_features)
        
        # Scale features if scaler exists
        if open_scaler is not None:
            X_open = open_scaler.transform(X_open)
        if close_scaler is not None:
            X_close = close_scaler.transform(X_close)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature creation error: {str(e)}"
        )
    
    # Make predictions
    try:
        predicted_open = float(open_model.predict(X_open)[0])
        predicted_close = float(close_model.predict(X_close)[0])
        change = predicted_close - predicted_open
        
        return {
            "city": city_key,
            "date": date,
            "predictions": {
                "open": round(predicted_open, 2),
                "close": round(predicted_close, 2),
                "expected_change": round(change, 2)
            },
            "currency": "PKR",
            "model_info": {
                "open": {
                    "type": open_data['type'],
                    "r2": round(open_data['r2'], 4),
                    "mae": round(open_data['mae'], 2)
                },
                "close": {
                    "type": close_data['type'],
                    "r2": round(close_data['r2'], 4),
                    "mae": round(close_data['mae'], 2)
                }
            },
            "note": "⚠️ Using simplified features. For production accuracy, integrate historical data."
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ## 🚀 **Deploy This Updated Code**

# 2. ✅ Push to GitHub
# 3. ✅ Render will auto-redeploy
# 4. ✅ Test with:
# ```
# https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-02-20&api_key=mysecretkey123
