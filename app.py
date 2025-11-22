from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import hashlib

app = FastAPI(title="Smart Poultry Prediction API")

API_KEY = "mysecretkey123"

# =====================================================
# LOAD CSV DATA - match predict.py
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
    raise Exception("❌ CSV file not found")

df = pd.read_csv(CSV_FILE, parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# --- Clean column names: strip spaces and remove internal spaces ---
df.columns = df.columns.str.strip().str.replace(" ", "")
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

cities = ['Rawalpindi', 'Lahore']

print(f"✅ Loaded CSV with {len(df)} rows. Latest date: {df['Date'].max()}")

# =====================================================
# LOAD MODELS - match predict.py
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
    raise Exception("❌ Model file not found")

master_models = {}
if os.path.exists(MODEL_FILE):
    print("🔄 Master joblib found! Loading Lahore + Rawalpindi models...")
    master_models = joblib.load(MODEL_FILE)

models = {}
print("\n🎯 LOADING/TRAINING MODELS PER CITY")

for city in cities:
    print(f"\n📍 {city}")
    print("-" * 60)
    for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
        if city in master_models and price_type in master_models[city]:
            print(f"   ✅ Loaded {city} {price_type} from master file")
            entry = master_models[city][price_type]
            # store with lowercase keys like predict.py
            models.setdefault(city, {})[price_type.lower()] = entry
            print(f"   {price_type}: {entry['type'].upper()} | R2={entry['r2']:.4f} | MAE={entry['mae']:.2f}")

print("\n✅ Models ready. You can start predictions now.")
print("=" * 70)

# =====================================================
# PREDICTION FUNCTIONS - match predict.py
# =====================================================
def load_model_entry(entry):
    model = entry.get('model', None)
    scaler = entry.get('scaler', None)
    return model, scaler

def compute_features_from_series(city, price_type, values_series, date):
    features = {}
    # Lag features - use most recent values
    for lag in range(1, 8):
        if lag < len(values_series):
            features[f'lag_{lag}'] = values_series[-lag-1]
        else:
            features[f'lag_{lag}'] = values_series[-1]

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

def predict_future_prices(city, target_date):
    # deterministic seed identical to predict.py
    seed_str = f"{city}_{target_date.strftime('%Y-%m-%d')}"
    seed_value = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed_value)

    if city not in models:
        return None

    # historical check
    historical = df[df['Date'] == target_date]
    if not historical.empty and pd.notna(historical.iloc[0][f'{city}_Open']):
        row = historical.iloc[0]
        result = {'type': 'historical', 'open': row[f'{city}_Open'], 'close': row[f'{city}_Close']}
        if f'{city}_FarmRate' in row and pd.notna(row[f'{city}_FarmRate']):
            result['farmrate'] = row[f'{city}_FarmRate']
        if f'{city}_DOC' in row and pd.notna(row[f'{city}_DOC']):
            result['doc'] = row[f'{city}_DOC']
        return result

    result = {'type': 'future'}
    latest_date = df['Date'].max()
    days_ahead = (target_date - latest_date).days

    # handle target_date before or equal latest_date (sensible behavior)
    if days_ahead <= 0:
        # if same day and no historical value, still attempt to return latest values
        latest_row = df[df['Date'] == latest_date].iloc[0]
        result['type'] = 'historical'
        result['open'] = latest_row[f'{city}_Open']
        result['close'] = latest_row[f'{city}_Close']
        if f'{city}_FarmRate' in latest_row and pd.notna(latest_row[f'{city}_FarmRate']):
            result['farmrate'] = latest_row[f'{city}_FarmRate']
        if f'{city}_DOC' in latest_row and pd.notna(latest_row[f'{city}_DOC']):
            result['doc'] = latest_row[f'{city}_DOC']
        return result

    # Get latest row and latest values
    latest_row = df[df['Date'] == latest_date].iloc[0]
    result['latest_date'] = latest_date
    result['latest_open'] = latest_row[f'{city}_Open']
    result['latest_close'] = latest_row[f'{city}_Close']
    if f'{city}_FarmRate' in latest_row and pd.notna(latest_row[f'{city}_FarmRate']):
        result['latest_farmrate'] = latest_row[f'{city}_FarmRate']
    if f'{city}_DOC' in latest_row and pd.notna(latest_row[f'{city}_DOC']):
        result['latest_doc'] = latest_row[f'{city}_DOC']

    # Get historical data for feature computation - USE last 50 values exactly
    historical_data = {}
    for price_type in ['Open', 'Close', 'FarmRate', 'DOC']:
        col = f'{city}_{price_type}'
        if col in df.columns:
            # Use last 50 values (dropna) - this was the mismatch fix
            hist_vals = df[col].dropna().values.tolist()[-50:]
            historical_data[price_type] = hist_vals

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

            values_series = predictions[price_type]
            feat_dict = compute_features_from_series(city, price_type, values_series, current_date)

            # Cross-city features (use latest available)
            for other_city in [c for c in cities if c != city]:
                col_name = f'{other_city}_lag1'
                if col_name in entry.get('features', []):
                    other_col = f'{other_city}_{price_type}'
                    if other_col in df.columns:
                        feat_dict[col_name] = df[other_col].dropna().iloc[-1]

            X_dict = {feat: feat_dict.get(feat, 0) for feat in entry['features']}
            X = pd.DataFrame([X_dict])[entry['features']].values

            model, scaler = load_model_entry(entry)
            if entry['type'] == 'mlp' and scaler is not None:
                X = scaler.transform(X)

            base_prediction = model.predict(X)[0]

            # volatility from last 7 predicted/actual values
            recent_vals = values_series[-7:] if len(values_series) >= 7 else values_series
            volatility = np.std(recent_vals) if len(recent_vals) > 1 else entry['mae']

            # noise multiplier is 0.3 as in your predict.py
            noise = np.random.normal(0, volatility * 0.3)
            prediction = base_prediction + noise

            max_deviation = entry['mae'] * 2
            prediction = np.clip(prediction, base_prediction - max_deviation, base_prediction + max_deviation)
            prediction = max(prediction, 0)

            predictions[price_type].append(prediction)

            if day == days_ahead:
                result[pt] = prediction
                result[f'{pt}_model_type'] = entry['type']
                result[f'{pt}_model_r2'] = entry['r2']
                result[f'{pt}_model_mae'] = entry['mae']

    return result

# =====================================================
# FLATTENED VS-CODE STYLE TEXT BUILDER
# =====================================================
def build_vscode_text(city, date_str, res):
    """
    Build the exact VS Code style multiline text and return it.
    Uses 2 decimal formatting and same lines as predict.py.
    """
    lines = []
    if res['type'] == 'historical':
        lines.append(f"📅 HISTORICAL DATA - {city} ({date_str})")
        lines.append(f"  Open:  Rs {res['open']:.2f}")
        lines.append(f"  Close: Rs {res['close']:.2f}")
        if 'farmrate' in res:
            lines.append(f"  FarmRate: Rs {res['farmrate']:.2f}")
        if 'doc' in res:
            lines.append(f"  DOC: Rs {res['doc']:.2f}")
        lines.append(f"  Daily Change: Rs {res['close'] - res['open']:+.2f}")
    else:
        lines.append(f"🔮 FUTURE PREDICTION - {city} ({date_str})")
        lines.append(f"  Predicted Open:  Rs {res['open']:.2f}")
        lines.append(f"  Predicted Close: Rs {res['close']:.2f}")
        if 'farmrate' in res:
            lines.append(f"  Predicted FarmRate: Rs {res['farmrate']:.2f}")
        if 'doc' in res:
            lines.append(f"  Predicted DOC: Rs {res['doc']:.2f}")
        lines.append("")  # blank line
        # Expected changes from latest
        latest_date_str = res['latest_date'].strftime('%Y-%m-%d') if 'latest_date' in res else ''
        lines.append(f"  📈 Expected Changes from Latest ({latest_date_str}):")
        open_change = res['open'] - res['latest_open']
        lines.append(f"  Open:  {open_change:+.2f} (±{res['open_model_mae']:.2f}) → Rs {res['latest_open']:.2f} → Rs {res['open']:.2f}")
        close_change = res['close'] - res['latest_close']
        lines.append(f"  Close: {close_change:+.2f} (±{res['close_model_mae']:.2f}) → Rs {res['latest_close']:.2f} → Rs {res['close']:.2f}")
        if 'farmrate' in res and 'latest_farmrate' in res:
            farmrate_change = res['farmrate'] - res['latest_farmrate']
            lines.append(f"  FarmRate: {farmrate_change:+.2f} (±{res['farmrate_model_mae']:.2f}) → Rs {res['latest_farmrate']:.2f} → Rs {res['farmrate']:.2f}")
        if 'doc' in res and 'latest_doc' in res:
            doc_change = res['doc'] - res['latest_doc']
            lines.append(f"  DOC: {doc_change:+.2f} (±{res['doc_model_mae']:.2f}) → Rs {res['latest_doc']:.2f} → Rs {res['doc']:.2f}")

    # join with newlines and keep same separators shown in VS Code printout
    return "\n".join(lines)

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

    # case-insensitive match
    city_key = None
    for key in cities:
        if key.lower() == city.lower():
            city_key = key
            break

    if not city_key:
        raise HTTPException(status_code=404, detail=f"City not found. Available: {cities}")

    try:
        target_date = pd.to_datetime(date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        res = predict_future_prices(city_key, target_date)
        if res is None:
            raise HTTPException(status_code=500, detail="Prediction error")

        # Prepare response body (structured + text)
        if res['type'] == 'historical':
            result_struct = {
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
                result_struct['predictions']['farmrate'] = round(float(res['farmrate']), 2)
            if 'doc' in res:
                result_struct['predictions']['doc'] = round(float(res['doc']), 2)

            # Build VS Code style text
            result_text = build_vscode_text(city_key, date, res)

            result_struct['result'] = result_text
            return result_struct

        else:
            # future
            result_struct = {
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

            if 'latest_farmrate' in res:
                result_struct['latest_data']['farmrate'] = round(float(res['latest_farmrate']), 2)
            if 'latest_doc' in res:
                result_struct['latest_data']['doc'] = round(float(res['latest_doc']), 2)

            for price_type in ['open', 'close', 'farmrate', 'doc']:
                if price_type in res:
                    result_struct['predictions'][price_type] = round(float(res[price_type]), 2)
                    result_struct['model_info'][price_type] = {
                        "type": res[f'{price_type}_model_type'],
                        "r2": round(float(res[f'{price_type}_model_r2']), 4),
                        "mae": round(float(res[f'{price_type}_model_mae']), 2)
                    }
                    latest_key = f'latest_{price_type}'
                    if latest_key in res:
                        change = res[price_type] - res[latest_key]
                        mae = res[f'{price_type}_model_mae']
                        result_struct['expected_changes'][price_type] = {
                            "change": round(float(change), 2),
                            "mae": round(float(mae), 2),
                            "from": round(float(res[latest_key]), 2),
                            "to": round(float(res[price_type]), 2)
                        }

            # expected change open->close
            if 'open' in result_struct['predictions'] and 'close' in result_struct['predictions']:
                result_struct['predictions']['expected_change'] = round(
                    result_struct['predictions']['close'] - result_struct['predictions']['open'], 2
                )

            # Build VS Code style text
            result_text = build_vscode_text(city_key, date, res)
            result_struct['result'] = result_text

            return result_struct

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
