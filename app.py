from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# NEW: Add this import for deterministic hashing
import hashlib

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# LOAD CSV DATA - EXACTLY LIKE predict.py LINE BY LINE
# =====================================================
CSV_PATHS = [
    "agbro_combined_cleaned.csv",
    "data/agbro_combined_cleaned.csv",
    "./agbro_combined_cleaned.csv",
    "./data/agbro_combined_cleaned.csv"
]

CSV_FILE = None
for path in CSV_PATHS:
    if os.path.exists(path):
        CSV_FILE = path
        break

if CSV_FILE is None:
    raise Exception(f"❌ CSV file not found")

# EXACT COPY from predict.py lines 12-18
df = pd.read_csv(CSV_FILE, parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# --- Clean column names: strip spaces and remove internal spaces ---
df.columns = df.columns.str.strip().str.replace(" ", "")
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

cities = ['Rawalpindi', 'Lahore']

print(f"✅ Loaded CSV with {len(df)} rows. Latest date: {df['Date'].max()}")

# =====================================================
# LOAD MODELS - EXACTLY LIKE predict.py
# =====================================================
MODEL_PATHS = [
    "lahore_rawalpindi_models.joblib",
    "models_selected/lahore_rawalpindi_models.joblib",
    "./lahore_rawalpindi_models.joblib",
    "./models_selected/lahore_rawalpindi_models.joblib"
]

MODEL_FILE = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_FILE = path
        break

if MODEL_FILE is None:
    raise Exception(f"❌ Model file not found")

# Load master_models EXACTLY like predict.py line 56-59
master_models = {}
if os.path.exists(MODEL_FILE):
    print("🔄 Master joblib found! Loading Lahore + Rawalpindi models...")
    master_models = joblib.load(MODEL_FILE)

# Build models dict EXACTLY like predict.py lines 89-100
models = {}
print("\n🎯 LOADING/TRAINING MODELS PER CITY")

for city in cities:
    print(f"\n📍 {city}")
    print("-" * 60)
    
    for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
        # Check if model is already saved in master joblib
        if city in master_models and price_type in master_models[city]:
            print(f"   ✅ Loaded {city} {price_type} from master file")
            entry = master_models[city][price_type]
            # CRITICAL: Store with LOWERCASE key like predict.py line 95
            models.setdefault(city, {})[price_type.lower()] = entry
            print(f"   {price_type}: {entry['type'].upper()} | R2={entry['r2']:.4f} | MAE={entry['mae']:.2f}")

print("\n✅ Models ready. You can start predictions now.")
print("=" * 70)

# =====================================================
# PREDICTION FUNCTIONS - EXACT COPY FROM predict.py
# =====================================================

# EXACT COPY from predict.py lines 142-145
def load_model_entry(entry):
    model = entry.get('model', None)
    scaler = entry.get('scaler', None)
    return model, scaler

# EXACT COPY from predict.py lines 147-217
def compute_features_from_series(city, price_type, values_series, date):
    """
    Compute all required features from a series of recent values
    values_series: list/array of recent prices (at least 30 values)
    """
    features = {}
    
    # Lag features - use most recent values
    for lag in range(1, 8):
        if lag < len(values_series):
            features[f'lag_{lag}'] = values_series[-lag-1]
        else:
            features[f'lag_{lag}'] = values_series[-1]  # Use last available instead of first
    
    # Moving averages
    for window in [3, 7, 14, 30]:
        if len(values_series) >= window:
            features[f'ma_{window}'] = np.mean(values_series[-window:])
            features[f'std_{window}'] = np.std(values_series[-window:])
        else:
            features[f'ma_{window}'] = np.mean(values_series)
            features[f'std_{window}'] = np.std(values_series)
    
    # EMA - weighted towards recent values
    if len(values_series) >= 7:
        ema_7 = pd.Series(values_series).ewm(span=7, adjust=False).mean().iloc[-1]
        features['ema_7'] = ema_7
    else:
        features['ema_7'] = values_series[-1]
    
    if len(values_series) >= 14:
        ema_14 = pd.Series(values_series).ewm(span=14, adjust=False).mean().iloc[-1]
        features['ema_14'] = ema_14
    else:
        features['ema_14'] = values_series[-1]
    
    # Percentage changes
    if len(values_series) >= 2:
        features['pct_change_1'] = (values_series[-1] - values_series[-2]) / values_series[-2] if values_series[-2] != 0 else 0
    else:
        features['pct_change_1'] = 0
    
    if len(values_series) >= 8:
        features['pct_change_7'] = (values_series[-1] - values_series[-8]) / values_series[-8] if values_series[-8] != 0 else 0
    else:
        features['pct_change_7'] = 0
    
    # Momentum
    if len(values_series) >= 4:
        features['momentum_3'] = values_series[-1] - values_series[-4]
    else:
        features['momentum_3'] = 0
    
    if len(values_series) >= 8:
        features['momentum_7'] = values_series[-1] - values_series[-8]
    else:
        features['momentum_7'] = 0
    
    # Volatility (using high_low_diff from Open/Close)
    features['high_low_diff'] = 0
    features['volatility_7'] = 0
    
    # Time features
    features['day_of_week'] = date.dayofweek
    features['day_of_month'] = date.day
    features['month'] = date.month
    features['quarter'] = date.quarter
    
    # Trend features
    if len(values_series) >= 7:
        features['trend_7'] = np.polyfit(range(7), values_series[-7:], 1)[0]
    else:
        features['trend_7'] = 0
    
    if len(values_series) >= 14:
        features['trend_14'] = np.polyfit(range(14), values_series[-14:], 1)[0]
    else:
        features['trend_14'] = 0
    
    return features

# CORRECTED: Now uses the EXACT deterministic seed from predict.py
def predict_future_prices(city, target_date):
    # 🔥 FIXED: Use deterministic hash for consistent seeds across runs (matches predict.py)
    seed_str = f"{city}_{target_date.strftime('%Y-%m-%d')}"
    seed_value = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed_value)
    
    if city not in models:
        return None
    
    # Check if historical data exists
    historical = df[df['Date'] == target_date]
    if not historical.empty and pd.notna(historical.iloc[0][f'{city}_Open']):
        row = historical.iloc[0]
        result = {'type': 'historical', 'open': row[f'{city}_Open'], 'close': row[f'{city}_Close']}
        if f'{city}_FarmRate' in row and pd.notna(row[f'{city}_FarmRate']):
            result['farmrate'] = row[f'{city}_FarmRate']
        if f'{city}_DOC' in row and pd.notna(row[f'{city}_DOC']):
            result['doc'] = row[f'{city}_DOC']
        return result
    
    # ITERATIVE MULTI-STEP FORECASTING
    result = {'type': 'future'}
    latest_date = df['Date'].max()
    days_ahead = (target_date - latest_date).days
    
    # Get the actual latest row first
    latest_row = df[df['Date'] == latest_date].iloc[0]
    
    result['latest_date'] = latest_date
    result['latest_open'] = latest_row[f'{city}_Open']
    result['latest_close'] = latest_row[f'{city}_Close']
    if f'{city}_FarmRate' in latest_row and pd.notna(latest_row[f'{city}_FarmRate']):
        result['latest_farmrate'] = latest_row[f'{city}_FarmRate']
    if f'{city}_DOC' in latest_row and pd.notna(latest_row[f'{city}_DOC']):
        result['latest_doc'] = latest_row[f'{city}_DOC']
    
    # Get historical data for feature computation - use last 49 values + append the actual latest
    historical_data = {}
    for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
        col = f'{city}_{price_type}'
        if col in df.columns:
            # Get last 49 historical values, then append the TRUE latest value
            hist_vals = df[col].iloc[:-1].dropna().values.tolist()[-49:]
            actual_latest = latest_row[col]
            if pd.notna(actual_latest):
                hist_vals.append(actual_latest)
            historical_data[price_type] = hist_vals
    
    # Predict iteratively for each day
    predictions = {
        'Open': historical_data.get('Open', []).copy(),
        'Close': historical_data.get('Close', []).copy(),
        'FarmRate': historical_data.get('FarmRate', []).copy(),
        'DOC': historical_data.get('DOC', []).copy()
    }
    
    current_date = latest_date
    for day in range(1, days_ahead + 1):
        current_date = current_date + pd.Timedelta(days=1)
        
        for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
            pt = price_type.lower()
            if pt not in models[city]:
                continue
            
            entry = models[city][pt]
            
            # Build features from accumulated predictions
            values_series = predictions[price_type]
            feat_dict = compute_features_from_series(city, price_type, values_series, current_date)
            
            # Cross-city features (use latest available)
            for other_city in [c for c in cities if c != city]:
                col_name = f'{other_city}_lag1'
                if col_name in entry['features']:
                    other_col = f'{other_city}_{price_type}'
                    if other_col in df.columns:
                        feat_dict[col_name] = df[other_col].dropna().iloc[-1]
            
            # Ensure all features are present
            X_dict = {feat: feat_dict.get(feat, 0) for feat in entry['features']}
            X = pd.DataFrame([X_dict])[entry['features']].values
            
            model, scaler = load_model_entry(entry)
            if entry['type'] == 'mlp' and scaler is not None:
                X = scaler.transform(X)
            
            base_prediction = model.predict(X)[0]
            
            # Add realistic market noise based on recent volatility
            # Calculate volatility from last 7 days
            recent_vals = values_series[-7:] if len(values_series) >= 7 else values_series
            volatility = np.std(recent_vals) if len(recent_vals) > 1 else entry['mae']
            
            # Add random noise proportional to volatility (±0.5 * volatility)
            # This simulates natural market fluctuations
            noise = np.random.normal(0, volatility * 0.3)
            prediction = base_prediction + noise
            
            # Ensure prediction doesn't deviate too much from recent trend
            # Keep within ±2*MAE of base prediction
            max_deviation = entry['mae'] * 2
            prediction = np.clip(prediction, base_prediction - max_deviation, base_prediction + max_deviation)
            
            # Ensure non-negative prices
            prediction = max(prediction, 0)
            
            predictions[price_type].append(prediction)
            
            # Store only final prediction
            if day == days_ahead:
                result[pt] = prediction
                result[f'{pt}_model_type'] = entry['type']
                result[f'{pt}_model_r2'] = entry['r2']
                result[f'{pt}_model_mae'] = entry['mae']
    
    return result

# =====================================================
# FASTAPI ROUTES
# =====================================================
@app.get("/")
def home():
    latest_data = {}
    for city in cities:
        latest_row = df.iloc[-1]
        city_data = {
            "date": latest_row['Date'].strftime('%Y-%m-%d'),
            "open": float(latest_row[f'{city}_Open']),
            "close": float(latest_row[f'{city}_Close'])
        }
        
        farmrate_col = f'{city}_FarmRate'
        if farmrate_col in df.columns and pd.notna(latest_row[farmrate_col]):
            city_data['farmrate'] = float(latest_row[farmrate_col])
        
        doc_col = f'{city}_DOC'
        if doc_col in df.columns and pd.notna(latest_row[doc_col]):
            city_data['doc'] = float(latest_row[doc_col])
        
        latest_data[city] = city_data
    
    return {
        "message": "🐔 Smart Poultry Prediction API - Exact Match with predict.py",
        "available_cities": cities,
        "latest_prices": latest_data,
        "csv_rows": len(df),
        "csv_latest_date": df['Date'].max().strftime('%Y-%m-%d'),
        "models_loaded": list(models.keys()),
        "note": "Uses identical prediction logic from predict.py",
        "example": "/predict_date?city=Rawalpindi&date=2025-11-29&api_key=mysecretkey123"
    }

@app.get("/predict_date")
def predict_date(city: str, date: str, api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Find matching city (case insensitive)
    city_key = None
    for key in cities:
        if key.lower() == city.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(status_code=404, detail=f"City not found. Available: {cities}")
    
    try:
        target_date = pd.to_datetime(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        # Call EXACT same function as predict.py
        res = predict_future_prices(city_key, target_date)
        
        if res is None:
            raise HTTPException(status_code=500, detail="Prediction error")
        
        if res['type'] == 'historical':
            result = {
                "city": city_key,
                "date": date,
                "type": "historical",
                "predictions": {
                    "open": round(float(res['open']), 2),
                    "close": round(float(res['close']), 2),
                    "expected_change": round(float(res['close'] - res['open']), 2)
                },
                "currency": "PKR"
            }
            
            if 'farmrate' in res:
                result['predictions']['farmrate'] = round(float(res['farmrate']), 2)
            if 'doc' in res:
                result['predictions']['doc'] = round(float(res['doc']), 2)
            
            return result
        
        else:  # future prediction - format like predict.py output
            result = {
                "city": city_key,
                "date": date,
                "type": "future",
                "predictions": {},
                "currency": "PKR",
                "latest_data": {
                    "date": res['latest_date'].strftime('%Y-%m-%d'),
                    "open": round(float(res['latest_open']), 2),
                    "close": round(float(res['latest_close']), 2)
                },
                "model_info": {},
                "expected_changes": {}
            }
            
            # Add latest farmrate and doc
            if 'latest_farmrate' in res:
                result['latest_data']['farmrate'] = round(float(res['latest_farmrate']), 2)
            if 'latest_doc' in res:
                result['latest_data']['doc'] = round(float(res['latest_doc']), 2)
            
            # Add predictions for all price types
            for price_type in ['open', 'close', 'farmrate', 'doc']:
                if price_type in res:
                    result['predictions'][price_type] = round(float(res[price_type]), 2)
                    
                    # Add model info
                    result['model_info'][price_type] = {
                        "type": res[f'{price_type}_model_type'],
                        "r2": round(float(res[f'{price_type}_model_r2']), 4),
                        "mae": round(float(res[f'{price_type}_model_mae']), 2)
                    
                    }
                    
                    # Calculate expected changes
                    latest_key = f'latest_{price_type}'
                    if latest_key in res:
                        change = res[price_type] - res[latest_key]
                        mae = res[f'{price_type}_model_mae']
                        result['expected_changes'][price_type] = {
                            "change": round(float(change), 2),
                            "mae": round(float(mae), 2),
                            "from": round(float(res[latest_key]), 2),
                            "to": round(float(res[price_type]), 2)
                        }
            
            # Expected change for Open to Close
            if 'open' in result['predictions'] and 'close' in result['predictions']:
                result['predictions']['expected_change'] = round(
                    result['predictions']['close'] - result['predictions']['open'], 2
                )
            
            return result
    
    except Exception as e:
        import traceback
       

