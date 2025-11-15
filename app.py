# app.py
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Hybrid Poultry Price Predictor", version="1.0")

# ---------------- Data & Model Paths ----------------
DATA_FILE = "agbro_combined_cleaned.csv"
MODEL_FILE = "lahore_rawalpindi_models.pkl"

# ---------------- Load Data ----------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in repo root!")
df = pd.read_csv(DATA_FILE, parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

# ---------------- Load Models ----------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found in repo root!")
models = joblib.load(MODEL_FILE)  # master_models dictionary

cities = list(models.keys())

# ---------------- Pydantic Request Model ----------------
class PredictRequest(BaseModel):
    city: str
    date: str  # format: YYYY-MM-DD

# ---------------- Helper Functions ----------------
def create_advanced_features(df, city, price_type):
    """
    Create features for a given city and price type.
    Must match the features used during training.
    """
    data = df[['Date', f'{city}_Open', f'{city}_Close']].copy()
    price_col = f'{city}_{price_type}'
    
    # Lags
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data[price_col].shift(lag)
    # Rolling stats
    for window in [3, 7, 14, 30]:
        data[f'ma_{window}'] = data[price_col].rolling(window).mean()
        data[f'std_{window}'] = data[price_col].rolling(window).std()
    # EMA
    data['ema_7'] = data[price_col].ewm(span=7).mean()
    data['ema_14'] = data[price_col].ewm(span=14).mean()
    # Percent changes
    data['pct_change_1'] = data[price_col].pct_change(1)
    data['pct_change_7'] = data[price_col].pct_change(7)
    # Momentum
    data['momentum_3'] = data[price_col] - data[price_col].shift(3)
    data['momentum_7'] = data[price_col] - data[price_col].shift(7)
    # High-low diff & volatility
    data['high_low_diff'] = data[f'{city}_Close'] - data[f'{city}_Open']
    data['volatility_7'] = data['high_low_diff'].rolling(7).std()
    # Time features
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['day_of_month'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['quarter'] = data['Date'].dt.quarter
    # Trend
    data['trend_7'] = data[price_col].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==7 else np.nan)
    data['trend_14'] = data[price_col].rolling(14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==14 else np.nan)
    
    # Cross-city lag
    for other_city in [c for c in cities if c != city]:
        col = f'{other_city}_{price_type}'
        if col in df.columns:
            data[f'{other_city}_lag1'] = df[col].shift(1)
    
    data = data.iloc[30:].reset_index(drop=True)
    return data

def load_model_entry(entry):
    """Return model and scaler if present"""
    model = entry.get('model', None)
    scaler = entry.get('scaler', None)
    return model, scaler

def predict_city_price(city: str, target_date: pd.Timestamp):
    """Return prediction for a city at a specific date"""
    if city not in models:
        raise HTTPException(status_code=400, detail=f"City '{city}' not available")

    city_models = models[city]

    # Get latest data for features
    open_data = create_advanced_features(df, city, 'Open').iloc[-1]
    close_data = create_advanced_features(df, city, 'Close').iloc[-1]

    # Predict Open
    open_entry = city_models['Open']
    X_open = open_data[open_entry['features']].values.reshape(1, -1)
    open_model, open_scaler = load_model_entry(open_entry)
    X_open_scaled = open_scaler.transform(X_open) if open_entry['type']=='mlp' and open_scaler else X_open
    pred_open = open_model.predict(X_open_scaled)[0]

    # Predict Close
    close_entry = city_models['Close']
    X_close = close_data[close_entry['features']].values.reshape(1, -1)
    close_model, close_scaler = load_model_entry(close_entry)
    X_close_scaled = close_scaler.transform(X_close) if close_entry['type']=='mlp' and close_scaler else X_close
    pred_close = close_model.predict(X_close_scaled)[0]

    return {
        "city": city,
        "date": target_date.strftime('%Y-%m-%d'),
        "predictions": {
            "open": round(float(pred_open), 2),
            "close": round(float(pred_close), 2),
            "expected_change": round(float(pred_close - pred_open), 2)
        },
        "currency": "PKR",
        "latest_data": {
            "date": df['Date'].max().strftime('%Y-%m-%d'),
            "open": float(df[f'{city}_Open'].iloc[-1]),
            "close": float(df[f'{city}_Close'].iloc[-1])
        },
        "model_info": {
            "open": {
                "type": open_entry['type'],
                "r2": round(float(open_entry['r2']), 4),
                "mae": round(float(open_entry['mae']), 2)
            },
            "close": {
                "type": close_entry['type'],
                "r2": round(float(close_entry['r2']), 4),
                "mae": round(float(close_entry['mae']), 2)
            }
        }
    }

# ---------------- FastAPI Endpoint ----------------
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        target_date = pd.to_datetime(request.date)
    except:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    res = predict_city_price(request.city, target_date)
    return res

# ---------------- Root Endpoint ----------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Hybrid Poultry Price Predictor API! Use /predict endpoint."}
