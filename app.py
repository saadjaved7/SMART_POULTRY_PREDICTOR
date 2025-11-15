from fastapi import FastAPI, HTTPException
import joblib  # Changed from pickle to joblib
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# COMPLETE LATEST PRICES WITH ALL REQUIRED DATA
# =====================================================
LATEST_PRICES = {
    'Rawalpindi': {
        'date': '2025-11-15',
        'open': 370.00,
        'close': 362.50,
        'high': 372.00,
        'low': 360.00,
        # Lags (1-7)
        'lag_1': 370.00,
        'lag_2': 368.00,
        'lag_3': 365.00,
        'lag_4': 363.00,
        'lag_5': 362.00,
        'lag_6': 361.00,
        'lag_7': 360.00,
        # Moving averages
        'ma_3': 367.67,
        'ma_7': 365.5,
        'ma_14': 363.2,
        'ma_30': 361.5,
        # Standard deviations
        'std_3': 2.5,
        'std_7': 3.2,
        'std_14': 4.1,
        'std_30': 5.0,
        # EMAs
        'ema_7': 366.2,
        'ema_14': 364.0,
    },
    'Lahore': {
        'date': '2025-11-15',
        'open': 350.00,
        'close': 360.00,
        'high': 362.00,
        'low': 348.00,
        # Lags (1-7)
        'lag_1': 350.00,
        'lag_2': 348.00,
        'lag_3': 347.00,
        'lag_4': 346.00,
        'lag_5': 345.50,
        'lag_6': 345.00,
        'lag_7': 345.00,
        # Moving averages
        'ma_3': 348.33,
        'ma_7': 347.5,
        'ma_14': 345.8,
        'ma_30': 344.2,
        # Standard deviations
        'std_3': 1.5,
        'std_7': 2.1,
        'std_14': 3.2,
        'std_30': 4.5,
        # EMAs
        'ema_7': 348.2,
        'ema_14': 346.5,
    }
}

# =====================================================
# LOAD MODELS (USING JOBLIB)
# =====================================================
models = {}
model_file = "lahore_rawalpindi_models.joblib"  # Changed to .joblib

if not os.path.exists(model_file):
    raise Exception(f"❌ Model file not found: {model_file}")

try:
    models = joblib.load(model_file)  # Using joblib.load
    print(f"✅ Loaded models: {list(models.keys())}")
    
    # Print model details for debugging
    for city in models:
        print(f"\n📍 {city}")
        for target in ['Open', 'Close']:
            if target in models[city]:
                print(f"  {target}: {models[city][target]['type']} | R2: {models[city][target]['r2']:.4f}")
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# CREATE ALL 30 FEATURES IN CORRECT ORDER
# =====================================================
def create_features(date_str: str, city: str, feature_names: list):
    """
    Creates exactly the 30 features needed:
    ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 
     'ma_3', 'std_3', 'ma_7', 'std_7', 'ma_14', 'std_14', 'ma_30', 'std_30',
     'ema_7', 'ema_14', 'pct_change_1', 'pct_change_7', 'momentum_3', 
     'momentum_7', 'high_low_diff', 'volatility_7', 'day_of_week', 
     'day_of_month', 'month', 'quarter', 'trend_7', 'trend_14', 'Rawalpindi_lag1']
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    city_prices = LATEST_PRICES[city]
    
    # Get cross-city data
    if city == 'Lahore':
        cross_city_data = LATEST_PRICES['Rawalpindi']
    else:
        cross_city_data = LATEST_PRICES['Lahore']
    
    # Calculate all features
    features_dict = {
        # Lags
        'lag_1': city_prices['lag_1'],
        'lag_2': city_prices['lag_2'],
        'lag_3': city_prices['lag_3'],
        'lag_4': city_prices['lag_4'],
        'lag_5': city_prices['lag_5'],
        'lag_6': city_prices['lag_6'],
        'lag_7': city_prices['lag_7'],
        
        # Moving averages
        'ma_3': city_prices['ma_3'],
        'ma_7': city_prices['ma_7'],
        'ma_14': city_prices['ma_14'],
        'ma_30': city_prices['ma_30'],
        
        # Standard deviations
        'std_3': city_prices['std_3'],
        'std_7': city_prices['std_7'],
        'std_14': city_prices['std_14'],
        'std_30': city_prices['std_30'],
        
        # EMAs
        'ema_7': city_prices['ema_7'],
        'ema_14': city_prices['ema_14'],
        
        # Percentage changes
        'pct_change_1': ((city_prices['close'] - city_prices['lag_1']) / city_prices['lag_1']) * 100,
        'pct_change_7': ((city_prices['close'] - city_prices['lag_7']) / city_prices['lag_7']) * 100,
        
        # Momentum
        'momentum_3': ((city_prices['close'] - city_prices['lag_3']) / city_prices['lag_3']) * 100,
        'momentum_7': ((city_prices['close'] - city_prices['lag_7']) / city_prices['lag_7']) * 100,
        
        # High-low difference
        'high_low_diff': city_prices['high'] - city_prices['low'],
        
        # Volatility
        'volatility_7': city_prices['std_7'] / city_prices['ma_7'] * 100,
        
        # Time features
        'day_of_week': target_date.weekday(),
        'day_of_month': target_date.day,
        'month': target_date.month,
        'quarter': (target_date.month - 1) // 3 + 1,
        
        # Trends
        'trend_7': 1.0 if city_prices['close'] > city_prices['lag_7'] else 0.0,
        'trend_14': 1.0 if city_prices['ma_7'] > city_prices['ma_14'] else 0.0,
        
        # Cross-city lag (Rawalpindi_lag1 or Lahore_lag1)
        'Rawalpindi_lag1': cross_city_data['lag_1'],
        'Lahore_lag1': cross_city_data['lag_1'],
    }
    
    # Build feature array in exact order
    feature_array = []
    for fname in feature_names:
        if fname in features_dict:
            feature_array.append(features_dict[fname])
        else:
            # Handle alternative cross-city names
            if fname == 'Rawalpindi_lag1' and city == 'Rawalpindi':
                feature_array.append(cross_city_data['lag_1'])
            elif fname == 'Lahore_lag1' and city == 'Lahore':
                feature_array.append(cross_city_data['lag_1'])
            else:
                print(f"⚠️ Missing feature: {fname}, using 0.0")
                feature_array.append(0.0)
    
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
            "Rawalpindi": {
                "date": LATEST_PRICES['Rawalpindi']['date'],
                "open": LATEST_PRICES['Rawalpindi']['open'],
                "close": LATEST_PRICES['Rawalpindi']['close']
            },
            "Lahore": {
                "date": LATEST_PRICES['Lahore']['date'],
                "open": LATEST_PRICES['Lahore']['open'],
                "close": LATEST_PRICES['Lahore']['close']
            }
        },
        "model_info": {
            city: {
                "Open": models[city]['Open']['type'],
                "Close": models[city]['Close']['type']
            } for city in models.keys()
        },
        "example": "https://smart-poultry-predictor-6gca.onrender.com/predict_date?city=Lahore&date=2025-11-18&api_key=mysecretkey123"
    }

@app.get("/predict_date")
def predict_date(city: str, date: str, api_key: str):
    
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    city_key = None
    for key in models.keys():
        if key.lower() == city.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found. Available: {list(models.keys())}")
    
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']
        
        # Create features
        X_open = create_features(date, city_key, open_data['features'])
        X_close = create_features(date, city_key, close_data['features'])
        
        # Apply scaling
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


@app.get("/debug")
def debug_features(city: str, date: str, api_key: str):
    """Debug endpoint to check feature generation"""
    
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    city_key = None
    for key in models.keys():
        if key.lower() == city.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(status_code=404, detail=f"City not found")
    
    try:
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']
        
        X_open = create_features(date, city_key, open_data['features'])
        X_close = create_features(date, city_key, close_data['features'])
        
        return {
            "city": city_key,
            "date": date,
            "open_features": {
                "expected": open_data['features'],
                "count": len(open_data['features']),
                "values": X_open[0].tolist(),
                "first_5": dict(zip(open_data['features'][:5], X_open[0][:5].tolist()))
            },
            "close_features": {
                "expected": close_data['features'],
                "count": len(close_data['features']),
                "values": X_close[0].tolist(),
                "first_5": dict(zip(close_data['features'][:5], X_close[0][:5].tolist()))
            },
            "latest_prices": LATEST_PRICES[city_key]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")
