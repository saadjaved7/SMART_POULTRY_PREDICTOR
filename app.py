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
# LATEST PRICE DATA (Update these regularly)
# =====================================================
LATEST_PRICES = {
    'Rawalpindi': {
        'date': '2025-11-15',
        'open': 370.00,
        'close': 362.50,
        'lag_1': 370.00,
        'lag_2': 368.00,
        'lag_3': 365.00,
        'lag_7': 360.00,
        'lag_14': 355.00,
        'ma_7': 365.5,
        'ma_14': 363.2,
        'ema_7': 366.2,
        'ema_14': 364.0,
        'cross_city_open': 350.00,
        'cross_city_close': 360.00,
    },
    'Lahore': {
        'date': '2025-11-15',
        'open': 350.00,
        'close': 360.00,
        'lag_1': 350.00,
        'lag_2': 348.00,
        'lag_3': 347.00,
        'lag_7': 345.00,
        'lag_14': 340.00,
        'ma_7': 347.5,
        'ma_14': 345.8,
        'ema_7': 348.2,
        'ema_14': 346.5,
        'cross_city_open': 370.00,
        'cross_city_close': 362.50,
    }
}

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
    
    for city in models.keys():
        print(f"   {city}:")
        for price_type in models[city].keys():
            model_data = models[city][price_type]
            print(f"      {price_type} ({model_data['type']}): R²={model_data['r2']:.4f}, MAE={model_data['mae']:.2f}")
            
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# HELPER: CREATE FEATURES WITH REAL PRICES
# =====================================================
def create_features(date_str: str, city: str, feature_names: list):
    """
    Creates features using real latest prices for the given city.
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    latest_date = datetime.strptime(LATEST_PRICES[city]['date'], "%Y-%m-%d")
    
    days_diff = (target_date - latest_date).days
    
    day_of_week = target_date.weekday()
    month = target_date.month
    day = target_date.day
    
    city_prices = LATEST_PRICES[city]
    
    features_dict = {
        'day_of_week': day_of_week,
        'month': month,
        'day': day,
        'lag_1': city_prices['lag_1'],
        'lag_2': city_prices['lag_2'],
        'lag_3': city_prices['lag_3'],
        'lag_7': city_prices['lag_7'],
        'lag_14': city_prices['lag_14'],
        'ma_7': city_prices['ma_7'],
        'ma_14': city_prices['ma_14'],
        'ema_7': city_prices['ema_7'],
        'ema_14': city_prices['ema_14'],
        'momentum_7': (city_prices['close'] - city_prices['lag_7']) / city_prices['lag_7'] * 100,
        'momentum_14': (city_prices['close'] - city_prices['lag_14']) / city_prices['lag_14'] * 100,
        'volatility_7': abs(city_prices['open'] - city_prices['close']) / city_prices['close'] * 100,
        'volatility_14': abs(city_prices['ma_7'] - city_prices['ma_14']) / city_prices['ma_14'] * 100,
        'trend': 1.0 if city_prices['close'] > city_prices['lag_7'] else -1.0,
        'cross_city_lag_1': city_prices['cross_city_close'],
        'cross_city_lag_7': city_prices['cross_city_close'] - 5.0,
        'days_since_latest': days_diff,
    }
    
    feature_array = []
    for feature_name in feature_names:
        if feature_name in features_dict:
            feature_array.append(features_dict[feature_name])
        else:
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
            "latest_data": LATEST_PRICES[city]['date'],
            "latest_open": LATEST_PRICES[city]['open'],
            "latest_close": LATEST_PRICES[city]['close'],
            "models": {
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
        }
    
    return {
        "message": "🐔 Smart Poultry Prediction API",
        "available_cities": list(models.keys()),
        "city_info": model_info,
        "example_usage": "https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-11-18&api_key=mysecretkey123",
        "note": "Using real latest prices from 2025-11-15"
    }

@app.get("/predict_date")
def predict_date(city: str, date: str, api_key: str):
    """
    Predict poultry prices for a given city and date.
    
    Example: /predict_date?city=Lahore&date=2025-11-18&api_key=mysecretkey123
    """
    
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
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
    
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    
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
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    
    try:
        X_open = create_features(date, city_key, open_features)
        X_close = create_features(date, city_key, close_features)
        
        if open_scaler is not None:
            X_open = open_scaler.transform(X_open)
        if close_scaler is not None:
            X_close = close_scaler.transform(X_close)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature error: {str(e)}")
    
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
            "latest_data": {
                "date": LATEST_PRICES[city_key]['date'],
                "open": LATEST_PRICES[city_key]['open'],
                "close": LATEST_PRICES[city_key]['close']
            },
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
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ## 🚀 **Deploy & Test:**

# 1. ✅ Update `app.py` with code above
# 2. ✅ Push to GitHub
# 3. ✅ Test with:
# ```
# https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-11-18&api_key=mysecretkey123
