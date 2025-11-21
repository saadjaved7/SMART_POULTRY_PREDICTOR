import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ---------------- Load data ----------------
df = pd.read_csv("data/agbro_combined_cleaned.csv", parse_dates=["Date"], dayfirst=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)
df.ffill(inplace=True)

# cities = ['Rawalpindi', 'Lahore', 'Faisalabad', 'Multan']
cities = ['Rawalpindi', 'Lahore']   # Faisalabad, Multan commented as requested

print("="*70)
print("🐔 HYBRID CITY-WISE PREDICTOR (XGB / RF / MLP)")
print("="*70)
print("\n🎯 Features: Lag, MA, EMA, Momentum, Volatility, Trend, Cross-city lags, Time")
print("="*70)

# ---------------- Feature creator ----------------
def create_advanced_features(df, city, price_type):
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

# ---------------- Model params ----------------
xgb_params = {'n_estimators':600, 'learning_rate':0.01, 'max_depth':7, 'min_child_weight':1,
              'subsample':0.9,'colsample_bytree':0.9,'gamma':0,'reg_alpha':0.05,'reg_lambda':1,
              'random_state':42,'tree_method':'hist','verbosity':0}

rf_params = {'n_estimators':200,'max_depth':12,'min_samples_split':6,'min_samples_leaf':3,'random_state':42,'n_jobs':-1}

mlp_params = {'hidden_layer_sizes':(128,64),'activation':'relu','solver':'adam','max_iter':500,'random_state':42}

MODEL_DIR = "models_selected"
os.makedirs(MODEL_DIR, exist_ok=True)

# Master single-file (joblib) for Lahore + Rawalpindi
MASTER_MODEL_FILE = os.path.join(MODEL_DIR, "lahore_rawalpindi_models.joblib")
master_models = {}
if os.path.exists(MASTER_MODEL_FILE):
    print("🔄 Master joblib found! Loading Lahore + Rawalpindi models...")
    master_models = joblib.load(MASTER_MODEL_FILE)

# ---------------- Training functions ----------------
def train_xgb(X, y): return XGBRegressor(**xgb_params).fit(X, y)
def train_rf(X, y): return RandomForestRegressor(**rf_params).fit(X, y)
def train_mlp(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = MLPRegressor(**mlp_params).fit(Xs, y)
    return model, scaler

def evaluate_model(model, X_test, y_test, model_type='xgb', scaler=None):
    if model_type=='mlp' and scaler is not None: X_test = scaler.transform(X_test)
    preds = model.predict(X_test)
    return {'r2': r2_score(y_test, preds), 'mae': mean_absolute_error(y_test, preds), 'preds': preds}

# ---------------- Train or Load city-wise models ----------------
models = {}
print("\n🎯 LOADING/TRAINING MODELS PER CITY")

for city in cities:
    print(f"\n📍 {city}")
    print("-"*60)
    
    for price_type in ['Open','Close']:
        # Check if model is already saved in master joblib
        if city in master_models and price_type in master_models[city]:
            print(f"   ✅ Loaded {city} {price_type} from master file")

            entry = master_models[city][price_type]
            data = create_advanced_features(df, city, price_type)

            # Attach for predictions
            if price_type == "Open":
                models.setdefault(city, {})['open'], models[city]['open_data'] = entry, data
            else:
                models.setdefault(city, {})['close'], models[city]['close_data'] = entry, data

            print(f"   {price_type}: {entry['type'].upper()} | R2={entry['r2']} | MAE={entry['mae']}")
            continue
        
        # Else train models
        print(f"   Training models for {price_type}...")
        data = create_advanced_features(df, city, price_type)
        features = [c for c in data.columns if c not in ['Date', f'{city}_Open', f'{city}_Close']]
        X, y = data[features], data[f'{city}_{price_type}']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        
        best_res, best_model, best_scaler, best_type = None, None, None, None
        for attempt in range(5):
            xgb_model = train_xgb(X_train, y_train)
            rf_model = train_rf(X_train, y_train)
            mlp_model, mlp_scaler = train_mlp(X_train, y_train)
            
            candidates = [
                ('xgb', evaluate_model(xgb_model, X_test, y_test), xgb_model, None),
                ('rf', evaluate_model(rf_model, X_test, y_test), rf_model, None),
                ('mlp', evaluate_model(mlp_model, X_test, y_test, 'mlp', mlp_scaler), mlp_model, mlp_scaler)
            ]
            candidates_sorted = sorted(candidates, key=lambda x: (x[1]['r2'], -x[1]['mae']), reverse=True)
            top_type, top_res, top_model, top_scaler = candidates_sorted[0]
            
            if top_res['mae'] <= 2:
                best_res, best_model, best_scaler, best_type = top_res, top_model, top_scaler, top_type
                break
            if best_res is None or top_res['mae'] < best_res['mae']:
                best_res, best_model, best_scaler, best_type = top_res, top_model, top_scaler, top_type
        
        # Save into master_models dict (single file later)
        entry = {
            'type': best_type,
            'model': best_model,
            'scaler': best_scaler,
            'r2': best_res['r2'],
            'mae': best_res['mae'],
            'features': features
        }

        if city not in master_models:
            master_models[city] = {}

        master_models[city][price_type] = entry

        # Keep for immediate predictions
        if price_type == 'Open':
            models.setdefault(city, {})['open'], models[city]['open_data'] = entry, data
        else:
            models.setdefault(city, {})['close'], models[city]['close_data'] = entry, data
        
        print(f"   {price_type}: {best_type.upper()} | R2={best_res['r2']:.4f} | MAE={best_res['mae']:.2f}")

# Save single master joblib for Lahore + Rawalpindi
print("\n💾 Saving combined Lahore + Rawalpindi model file...")
joblib.dump(master_models, MASTER_MODEL_FILE)
print("✅ Saved as:", MASTER_MODEL_FILE)

print("\n✅ Models ready. You can start predictions now.")
print("="*70)

# ---------------- Prediction functions ----------------
def load_model_entry(entry):
    # entry already contains 'model' and possibly 'scaler'
    model = entry.get('model', None)
    scaler = entry.get('scaler', None)
    return model, scaler

def predict_future_prices(city, target_date):
    if city not in models: return None
    historical = df[df['Date'] == target_date]
    if not historical.empty and pd.notna(historical.iloc[0][f'{city}_Open']):
        row = historical.iloc[0]
        return {'type': 'historical', 'open': row[f'{city}_Open'], 'close': row[f'{city}_Close']}
    
    open_data = models[city]['open_data'].iloc[-1]
    close_data = models[city]['close_data'].iloc[-1]
    # Open prediction
    open_entry = models[city]['open']
    X_open = open_data[open_entry['features']].values.reshape(1, -1)
    open_model, open_scaler = load_model_entry(open_entry)
    pred_open = open_model.predict(open_scaler.transform(X_open) if open_entry['type'] == 'mlp' and open_scaler is not None else X_open)[0]
    # Close prediction
    close_entry = models[city]['close']
    X_close = close_data[close_entry['features']].values.reshape(1, -1)
    close_model, close_scaler = load_model_entry(close_entry)
    pred_close = close_model.predict(close_scaler.transform(X_close) if close_entry['type'] == 'mlp' and close_scaler is not None else X_close)[0]
    
    return {'type': 'future', 'open': pred_open, 'close': pred_close,
            'latest_open': open_data[f'{city}_Open'], 'latest_close': close_data[f'{city}_Close'],
            'latest_date': open_data['Date'],
            'open_model_type': open_entry['type'], 'open_model_r2': open_entry['r2'], 'open_model_mae': open_entry['mae'],
            'close_model_type': close_entry['type'], 'close_model_r2': close_entry['r2'], 'close_model_mae': close_entry['mae']}

# ---------------- Interactive prompt ----------------
print("\n💡 READY FOR PREDICTIONS")

while True:
    print(f"\nAvailable cities: {', '.join(cities)}")
    city = input("Enter city: ").strip()
    if city not in cities: print("❌ Invalid city"); continue
    date_str = input("Enter date (YYYY-MM-DD): ").strip()
    try: target_date = pd.to_datetime(date_str)
    except: print("❌ Invalid date"); continue
    res = predict_future_prices(city, target_date)
    if res is None: print("❌ Prediction error"); continue
    
    print("\n" + "="*50)
    if res['type'] == 'historical':
        print(f"📅 HISTORICAL DATA - {city} ({target_date.strftime('%Y-%m-%d')})")
        print(f"  Open: Rs {res['open']:.2f} | Close: Rs {res['close']:.2f} | Change: Rs {res['close']-res['open']:+.2f}")
    else:
        print(f"🔮 FUTURE PREDICTION - {city} ({target_date.strftime('%Y-%m-%d')})")
        print(f"  Predicted Open: Rs {res['open']:.2f}")
        print(f"  Predicted Close: Rs {res['close']:.2f}")
        
        # Calculate changes from LATEST prices
        open_change = res['open'] - res['latest_open']
        close_change = res['close'] - res['latest_close']
        
        print(f"\n  📈 Expected Changes from Latest ({res['latest_date'].strftime('%Y-%m-%d')}):")
        print(f"  Open:  {open_change:+.2f} (±{res['open_model_mae']:.2f}) → Rs {res['latest_open']:.2f} → Rs {res['open']:.2f}")
        print(f"  Close: {close_change:+.2f} (±{res['close_model_mae']:.2f}) → Rs {res['latest_close']:.2f} → Rs {res['close']:.2f}")
        
        print(f"\n  Latest Data ({res['latest_date'].strftime('%Y-%m-%d')}): Open: Rs {res['latest_open']:.2f} | Close: Rs {res['latest_close']:.2f}")
        print("\n  📊 Selected Models & Accuracy:")
        open_r2 = f"{res['open_model_r2']:.4f}" if res['open_model_r2'] is not None else "N/A"
        open_mae = f"Rs {res['open_model_mae']:.2f}" if res['open_model_mae'] is not None else "N/A"
        close_r2 = f"{res['close_model_r2']:.4f}" if res['close_model_r2'] is not None else "N/A"
        close_mae = f"Rs {res['close_model_mae']:.2f}" if res['close_model_mae'] is not None else "N/A"
        print(f"  Open  - {res['open_model_type'].upper()} | R2: {open_r2} | MAE: {open_mae}")
        print(f"  Close - {res['close_model_type'].upper()} | R2: {close_r2} | MAE: {close_mae}")
    print("="*50)
    
    if input("\nAnother prediction? (y/n): ").lower() != 'y': break

print("\n✅ Thank you for using the Hybrid Poultry Price Predictor!")
