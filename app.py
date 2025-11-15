from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# LOAD DATA
# =====================================================
DATA_FILE = "agbro_combined_cleaned.csv"

if not os.path.exists(DATA_FILE):
    raise Exception(f"❌ Data file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, parse_dates=['date'])

# =====================================================
# LOAD MODELS
# =====================================================
models = {}
model_file = "lahore_rawalpindi_models.pkl"

if not os.path.exists(model_file):
    raise Exception(f"❌ Model file not found: {model_file}")

try:
    with open(model_file, 'rb') as f:
        models = pickle.load(f)
    print(f"✅ Loaded models: {list(models.keys())}")
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# GET LATEST DATA FOR CITY
# =====================================================
def get_latest_prices(city):
    city_df = df[df['city'].str.lower() == city.lower()].sort_values('date')
    latest = city_df.iloc[-1]
    return latest.to_dict()

# =====================================================
# CREATE FEATURES
# =====================================================
def create_features(date_str: str, city: str, feature_names: list):
    """
    Create the 30 features required for the model
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    city_df = df[df['city'].str.lower() == city.lower()].sort_values('date')
    
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for city {city}")

    # Use last 30 days for moving averages, stds, etc.
    last_data = city_df.iloc[-1]
    last_7 = city_df.tail(7)
    last_14 = city_df.tail(14)
    last_30 = city_df.tail(30)

    # Cross-city lag1
    cross_city = df[df['city'].str.lower() != city.lower()].sort_values('date')
    cross_lag1 = cross_city['close'].iloc[-1] if not cross_city.empty else 0.0

    features_dict = {
        'lag_1': last_7['close'].iloc[-1],
        'lag_2': last_7['close'].iloc[-2] if len(last_7) >= 2 else last_7['close'].iloc[-1],
        'lag_3': last_7['close'].iloc[-3] if len(last_7) >= 3 else last_7['close'].iloc[-1],
        'lag_4': last_7['close'].iloc[-4] if len(last_7) >= 4 else last_7['close'].iloc[-1],
        'lag_5': last_7['close'].iloc[-5] if len(last_7) >= 5 else last_7['close'].iloc[-1],
        'lag_6': last_7['close'].iloc[-6] if len(last_7) >= 6 else last_7['close'].iloc[-1],
        'lag_7': last_7['close'].iloc[-7] if len(last_7) >= 7 else last_7['close'].iloc[-1],
        'ma_3': last_7['close'].tail(3).mean(),
        'ma_7': last_7['close'].mean(),
        'ma_14': last_14['close'].mean(),
        'ma_30': last_30['close'].mean(),
        'std_3': last_7['close'].tail(3).std(),
        'std_7': last_7['close'].std(),
        'std_14': last_14['close'].std(),
        'std_30': last_30['close'].std(),
        'ema_7': last_7['close'].ewm(span=7, adjust=False).mean().iloc[-1],
        'ema_14': last_14['close'].ewm(span=14, adjust=False).mean().iloc[-1],
        'pct_change_1': ((last_data['close'] - last_7['close'].iloc[-1]) / last_7['close'].iloc[-1]) * 100,
        'pct_change_7': ((last_data['close'] - last_7['close'].iloc[0]) / last_7['close'].iloc[0]) * 100,
        'momentum_3': ((last_data['close'] - last_7['close'].tail(3).iloc[0]) / last_7['close'].tail(3).iloc[0]) * 100,
        'momentum_7': ((last_data['close'] - last_7['close'].iloc[0]) / last_7['close'].iloc[0]) * 100,
        'high_low_diff': last_data['high'] - last_data['low'],
        'volatility_7': last_7['close'].std() / last_7['close'].mean() * 100,
        'day_of_week': target_date.weekday(),
        'day_of_month': target_date.day,
        'month': target_date.month,
        'quarter': (target_date.month - 1) // 3 + 1,
        'trend_7': 1.0 if last_7['close'].iloc[-1] > last_7['close'].iloc[0] else 0.0,
        'trend_14': 1.0 if last_14['close'].iloc[-1] > last_14['close'].iloc[0] else 0.0,
        'Rawalpindi_lag1': cross_lag1,
        'Lahore_lag1': cross_lag1
    }

    # Build array in exact feature order
    feature_array = [features_dict.get(f, 0.0) for f in feature_names]
    return np.array([feature_array])

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    return {
        "message": "🐔 Smart Poultry Prediction API",
        "available_cities": list(models.keys()),
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
        raise HTTPException(status_code=404, detail=f"City not found")

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']

        X_open = create_features(date, city_key, open_data['features'])
        X_close = create_features(date, city_key, close_data['features'])

        # Apply scaling
        if open_data['scaler']:
            X_open = open_data['scaler'].transform(X_open)
        if close_data['scaler']:
            X_close = close_data['scaler'].transform(X_close)

        predicted_open = float(open_data['model'].predict(X_open)[0])
        predicted_close = float(close_data['model'].predict(X_close)[0])

        latest_data = get_latest_prices(city_key)

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
                "date": str(latest_data['date'].date()) if isinstance(latest_data['date'], pd.Timestamp) else str(latest_data['date']),
                "open": float(latest_data['open']),
                "close": float(latest_data['close'])
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
