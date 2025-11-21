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
CSV_FILE = "agbro_combined_cleaned.csv"
CSV_PATHS = [
    "agbro_combined_cleaned.csv",
    "data/agbro_combined_cleaned.csv",
    "./agbro_combined_cleaned.csv",
    "./data/agbro_combined_cleaned.csv"
]

for path in CSV_PATHS:
    if os.path.exists(path):
        CSV_FILE = path
        break

if not os.path.exists(CSV_FILE):
    raise Exception(f"❌ CSV file not found: {CSV_FILE}")

# Load and clean data exactly like predict.py
df = pd.read_csv(CSV_FILE, parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
# Clean column names: strip spaces and remove internal spaces
df.columns = df.columns.str.strip().str.replace(" ", "")
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

print(f"✅ Loaded CSV with {len(df)} rows. Latest date: {df['Date'].max()}")
print(f"✅ Columns: {list(df.columns)}")

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
        for target in ['Open', 'Close', 'FarmRate', 'DOC']:
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
    price_col = f'{city}_{price_type}'
    if price_col not in df.columns:
        print(f"⚠️  Warning: {price_col} not found in dataset")
        return None
    
    # Start with necessary columns
    cols = ['Date', f'{city}_Open', f'{city}_Close']
    if price_type in ['FarmRate', 'DOC']:
        cols.append(price_col)
    data = df[cols].copy()
    
    # Lag features
    for lag in range(1, 8):
        data[f'lag_{lag}'] = data[price_col].shift(lag)
    
    # Moving averages and volatility
    for window in [3, 7, 14, 30]:
        data[f'ma_{window}'] = data[price_col].rolling(window).mean()
        data[f'std_{window}'] = data[price_col].rolling(window).std()
    
    # Exponential moving averages
    data['ema_7'] = data[price_col].ewm(span=7).mean()
    data['ema_14'] = data[price_col].ewm(span=14).mean()
    
    # Percentage changes
    data['pct_change_1'] = data[price_col].pct_change(1)
    data['pct_change_7'] = data[price_col].pct_change(7)
    
    # Momentum features
    data['momentum_3'] = data[price_col] - data[price_col].shift(3)
    data['momentum_7'] = data[price_col] - data[price_col].shift(7)
    
    # High-low difference and volatility
    data['high_low_diff'] = data[f'{city}_Close'] - data[f'{city}_Open']
    data['volatility_7'] = data['high_low_diff'].rolling(7).std()
    
    # Time features
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['day_of_month'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['quarter'] = data['Date'].dt.quarter
    
    # Trend features
    data['trend_7'] = data[price_col].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan
    )
    data['trend_14'] = data[price_col].rolling(14).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else np.nan
    )
    
    # Cross-city features
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
    feature_data[city] = {}
    for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
        features = create_advanced_features(df, city, price_type)
        if features is not None:
            feature_data[city][price_type] = features
            print(f"✅ Generated features for {city} - {price_type}")

# =====================================================
# ROUTES
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
        
        # Add FarmRate if available
        farmrate_col = f'{city}_FarmRate'
        if farmrate_col in df.columns and pd.notna(latest_row[farmrate_col]):
            city_data['farmrate'] = float(latest_row[farmrate_col])
        
        # Add DOC if available
        doc_col = f'{city}_DOC'
        if doc_col in df.columns and pd.notna(latest_row[doc_col]):
            city_data['doc'] = float(latest_row[doc_col])
        
        latest_data[city] = city_data
    
    return {
        "message": "🐔 Smart Poultry Prediction API (CSV-Powered with FarmRate & DOC)",
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
            result = {
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
            
            # Add FarmRate if available
            farmrate_col = f'{city_key}_FarmRate'
            if farmrate_col in df.columns and pd.notna(row[farmrate_col]):
                result['predictions']['farmrate'] = round(float(row[farmrate_col]), 2)
            
            # Add DOC if available
            doc_col = f'{city_key}_DOC'
            if doc_col in df.columns and pd.notna(row[doc_col]):
                result['predictions']['doc'] = round(float(row[doc_col]), 2)
            
            return result
        
        # Future prediction
        result = {
            "city": city_key,
            "date": date,
            "type": "future",
            "predictions": {},
            "currency": "PKR",
            "latest_data": {},
            "model_info": {}
        }
        
        # Get latest actual data
        latest_row = df.iloc[-1]
        latest_date = latest_row['Date']
        
        result['latest_data']['date'] = latest_date.strftime('%Y-%m-%d')
        result['latest_data']['open'] = round(float(latest_row[f'{city_key}_Open']), 2)
        result['latest_data']['close'] = round(float(latest_row[f'{city_key}_Close']), 2)
        
        # Predict for each price type: Open, Close, FarmRate, DOC
        for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
            if price_type not in models[city_key]:
                continue
            
            if price_type not in feature_data[city_key]:
                continue
            
            price_data = models[city_key][price_type]
            features_df = feature_data[city_key][price_type]
            
            # Get the last row (most recent data)
            feature_row = features_df.iloc[-1]
            
            # Extract features in correct order
            X = feature_row[price_data['features']].values.reshape(1, -1)
            
            # Convert to DataFrame with column names for scaler
            X_df = pd.DataFrame(X, columns=price_data['features'])
            
            # Apply scaling if needed
            if price_data['scaler']:
                X_scaled = price_data['scaler'].transform(X_df)
            else:
                X_scaled = X_df
            
            # Predict
            predicted_value = float(price_data['model'].predict(X_scaled)[0])
            
            # Add to result
            price_key = price_type.lower()
            result['predictions'][price_key] = round(predicted_value, 2)
            
            # Add model info
            result['model_info'][price_key] = {
                "type": price_data['type'],
                "r2": round(price_data['r2'], 4),
                "mae": round(price_data['mae'], 2)
            }
            
            # Add latest data for FarmRate and DOC
            price_col = f'{city_key}_{price_type}'
            if price_col in df.columns and pd.notna(latest_row[price_col]):
                result['latest_data'][price_key] = round(float(latest_row[price_col]), 2)
        
        # Calculate expected change for Open to Close
        if 'open' in result['predictions'] and 'close' in result['predictions']:
            result['predictions']['expected_change'] = round(
                result['predictions']['close'] - result['predictions']['open'], 2
            )
        
        return result
    
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
        debug_info = {
            "city": city_key,
            "date": date,
            "csv_info": {
                "total_rows": len(df),
                "latest_date": df['Date'].max().strftime('%Y-%m-%d'),
                "columns": list(df.columns)
            },
            "available_price_types": list(feature_data[city_key].keys()),
            "features_info": {}
        }
        
        for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
            if price_type not in models[city_key]:
                continue
            
            if price_type not in feature_data[city_key]:
                continue
            
            price_data = models[city_key][price_type]
            features_df = feature_data[city_key][price_type]
            feature_row = features_df.iloc[-1]
            
            debug_info['features_info'][price_type] = {
                "expected_features": price_data['features'],
                "count": len(price_data['features']),
                "first_5": dict(zip(
                    price_data['features'][:5], 
                    feature_row[price_data['features'][:5]].tolist()
                )),
                "latest_date": feature_row['Date'].strftime('%Y-%m-%d')
            }
        
        return debug_info
    
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}\n{traceback.format_exc()}")
