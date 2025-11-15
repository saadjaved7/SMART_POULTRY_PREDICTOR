from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# REAL LATEST PRICES (from your VS Code output)
# =====================================================
LATEST_PRICES = {
    'Rawalpindi': {
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
        'cross_city': 360.00,  # Lahore's close
    },
    'Lahore': {
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
        'cross_city': 362.50,  # Rawalpindi's close
    }
}

# =====================================================
# LOAD MODELS
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
# CREATE FEATURES WITH REAL PRICES
# =====================================================
def create_features(date_str: str, city: str, feature_names: list):
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    city_prices = LATEST_PRICES[city]
    
    features_dict = {
        'day_of_week': target_date.weekday(),
        'month': target_date.month,
        'day': target_date.day,
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
        'cross_city_lag_1': city_prices['cross_city'],
        'cross_city_lag_7': city_prices['cross_city'] - 5.0,
    }
    
    feature_array = [features_dict.get(name, 0.0) for name in feature_names]
    return np.array([feature_array])

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    return {
        "message": "🐔 Smart Poultry Prediction API",
        "available_cities": list(models.keys()),
        "latest_prices": {
            "Rawalpindi": f"Rs {LATEST_PRICES['Rawalpindi']['open']} - {LATEST_PRICES['Rawalpindi']['close']}",
            "Lahore": f"Rs {LATEST_PRICES['Lahore']['open']} - {LATEST_PRICES['Lahore']['close']}"
        },
        "example": "https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-11-18&api_key=mysecretkey123"
    }

@app.get("/predict_date")
def predict_date(city: str, date: str, api_key: str):
    
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Find city (case-insensitive)
    city_key = None
    for key in models.keys():
        if key.lower() == city.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(status_code=404, detail=f"City not found. Available: {list(models.keys())}")
    
    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Get models
    try:
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']
        
        # Create and scale features
        X_open = create_features(date, city_key, open_data['features'])
        X_close = create_features(date, city_key, close_data['features'])
        
        if open_data['scaler']:
            X_open = open_data['scaler'].transform(X_open)
        if close_data['scaler']:
            X_close = close_data['scaler'].transform(X_close)
        
        # Predict
        predicted_open = float(open_data['model'].predict(X_open)[0])
        predicted_close = float(close_data['model'].predict(X_close)[0])
        
        return {
            "city": city_key,
            "date": date,
            "predictions": {
                "open": round(predicted_open, 2),
                "close": round(predicted_close, 2),
                "expected_change": round(predicted_close - predicted_open, 2)
            },
            "currency": "PKR",
            "latest_data": {
                "date": "2025-11-15",
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
