from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# LOAD CSV DATA - EXACTLY LIKE predict.py
# =====================================================
CSV_FILE = "agbro_combined_cleaned.csv"  # Put this in your repo root
# Alternative paths to try
CSV_PATHS = [
    "agbro_combined_cleaned.csv",
    "data/agbro_combined_cleaned.csv",
    "./agbro_combined_cleaned.csv",
    "./data/agbro_combined_cleaned.csv"
]

# Try to find CSV file
for path in CSV_PATHS:
    if os.path.exists(path):
        CSV_FILE = path
        break

if not os.path.exists(CSV_FILE):
    raise Exception(f"❌ CSV file not found: {CSV_FILE}")

# Load data exactly like predict.py
df = pd.read_csv(CSV_FILE, parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

print(f"✅ Loaded CSV with {len(df)} rows. Latest date: {df['Date'].max()}")

# =====================================================
# LOAD MODELS
# =====================================================
cities = ['Rawalpindi', 'Lahore']
models = {}
model_file = "lahore_rawalpindi_models.joblib"

if not os.path.exists(model_file):
    raise Exception(f"❌ Model file not found: {model_file}")

try:
    models = joblib.load(model_file)
    print(f"✅ Loaded models: {list(models.keys())}")
    
    for city in models:
        print(f"\n📍 {city}")
        for target in ['Open', 'Close']:
            if target in models[city]:
                print(f"  {target}: {models[city][target]['type']} | R2: {models[city][target]['r2']:.4f}")
except Exception as e:
    raise Exception(f"❌ Failed to load models: {str(e)}")

# =====================================================
# FEATURE CREATOR - EXACT COPY FROM predict.py
# =====================================================
def create_advanced_features(df, city, price_type):
    """
    EXACT copy of the function from predict.py
    """
    data = df[['Date', f'{city}_Open', f'{city}_Close']].copy()
    price_col = f'{city}_{price_type}'
    
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data[price_col].shift(lag)
    for window in [3, 7, 14, 30]:
        data[f'ma_{window}'] = data[price_col].rolling(window).mean()
        data[f'std_{window}'] = data[price_col].rolling(window).std()
    data['ema_7'] = data[price_col].ewm(span=7).mean()
    data['ema_14'] = data[price_col].ewm(span=14).mean()
    data['pct_change_1'] = data[price_col].pct_change(1)
    data['pct_change_7'] = data[price_col].pct_change(7)
    data['momentum_3'] = data[price_col] - data[price_col].shift(3)
    data['momentum_7'] = data[price_col] - data[price_col].shift(7)
    data['high_low_diff'] = data[f'{city}_Close'] - data[f'{city}_Open']
    data['volatility_7'] = data['high_low_diff'].rolling(7).std()
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['day_of_month'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['quarter'] = data['Date'].dt.quarter
    data['trend_7'] = data[price_col].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan)
    data['trend_14'] = data[price_col].rolling(14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else np.nan)
    
    for other_city in [c for c in cities if c != city]:
        col = f'{other_city}_{price_type}'
        if col in df.columns:
            data[f'{other_city}_lag1'] = df[col].shift(1)
    
    data = data.iloc[30:].reset_index(drop=True)
    return data

# =====================================================
# PRE-GENERATE FEATURE DATA FOR EACH CITY
# =====================================================
feature_data = {}
for city in cities:
    feature_data[city] = {
        'Open': create_advanced_features(df, city, 'Open'),
        'Close': create_advanced_features(df, city, 'Close')
    }
    print(f"✅ Generated features for {city}")

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    latest_data = {}
    for city in cities:
        latest_row = df.iloc[-1]
        latest_data[city] = {
            "date": latest_row['Date'].strftime('%Y-%m-%d'),
            "open": float(latest_row[f'{city}_Open']),
            "close": float(latest_row[f'{city}_Close'])
        }
    
    return {
        "message": "🐔 Smart Poultry Prediction API (CSV-Powered)",
        "available_cities": list(models.keys()),
        "latest_prices": latest_data,
        "csv_loaded": True,
        "csv_rows": len(df),
        "csv_latest_date": df['Date'].max().strftime('%Y-%m-%d'),
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
        raise HTTPException(status_code=404, detail=f"City not found. Available: {list(models.keys())}")
    
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        # Check if historical data exists
        historical = df[df['Date'] == target_date]
        if not historical.empty and pd.notna(historical.iloc[0][f'{city_key}_Open']):
            row = historical.iloc[0]
            return {
                "city": city_key,
                "date": date,
                "type": "historical",
                "predictions": {
                    "open": round(float(row[f'{city_key}_Open']), 2),
                    "close": round(float(row[f'{city_key}_Close']), 2),
                    "expected_change": round(float(row[f'{city_key}_Close'] - row[f'{city_key}_Open']), 2)
                },
                "currency": "PKR"
            }
        
        # Future prediction
        open_data = models[city_key]['Open']
        close_data = models[city_key]['Close']
        
        # Get latest features from pre-generated data
        open_features_df = feature_data[city_key]['Open']
        close_features_df = feature_data[city_key]['Close']
        
        # Get the last row (most recent data)
        open_row = open_features_df.iloc[-1]
        close_row = close_features_df.iloc[-1]
        
        # Extract features in correct order
        X_open = open_row[open_data['features']].values.reshape(1, -1)
        X_close = close_row[close_data['features']].values.reshape(1, -1)
        
        # Convert to DataFrame with column names for scaler
        X_open_df = pd.DataFrame(X_open, columns=open_data['features'])
        X_close_df = pd.DataFrame(X_close, columns=close_data['features'])
        
        # Apply scaling
        if open_data['scaler']:
            X_open_scaled = open_data['scaler'].transform(X_open_df)
        else:
            X_open_scaled = X_open_df
            
        if close_data['scaler']:
            X_close_scaled = close_data['scaler'].transform(X_close_df)
        else:
            X_close_scaled = X_close_df
        
        # Predict
        predicted_open = float(open_data['model'].predict(X_open_scaled)[0])
        predicted_close = float(close_data['model'].predict(X_close_scaled)[0])
        
        # Get latest actual data
        latest_date = open_features_df.iloc[-1]['Date']
        latest_open = df[df['Date'] == latest_date].iloc[0][f'{city_key}_Open']
        latest_close = df[df['Date'] == latest_date].iloc[0][f'{city_key}_Close']
        
        return {
            "city": city_key,
            "date": date,
            "type": "future",
            "predictions": {
                "open": round(predicted_open, 2),
                "close": round(predicted_close, 2),
                "expected_change": round(predicted_close - predicted_open, 2)
            },
            "currency": "PKR",
            "latest_data": {
                "date": latest_date.strftime('%Y-%m-%d'),
                "open": round(float(latest_open), 2),
                "close": round(float(latest_close), 2)
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
        import traceback
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{traceback.format_exc()}")


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
        
        open_features_df = feature_data[city_key]['Open']
        close_features_df = feature_data[city_key]['Close']
        
        open_row = open_features_df.iloc[-1]
        close_row = close_features_df.iloc[-1]
        
        return {
            "city": city_key,
            "date": date,
            "open_features": {
                "expected": open_data['features'],
                "count": len(open_data['features']),
                "values": open_row[open_data['features']].tolist(),
                "first_5": dict(zip(open_data['features'][:5], open_row[open_data['features'][:5]].tolist()))
            },
            "close_features": {
                "expected": close_data['features'],
                "count": len(close_data['features']),
                "values": close_row[close_data['features']].tolist(),
                "first_5": dict(zip(close_data['features'][:5], close_row[close_data['features'][:5]].tolist()))
            },
            "latest_date": open_row['Date'].strftime('%Y-%m-%d'),
            "csv_info": {
                "total_rows": len(df),
                "latest_date": df['Date'].max().strftime('%Y-%m-%d')
            }
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}\n{traceback.format_exc()}")
