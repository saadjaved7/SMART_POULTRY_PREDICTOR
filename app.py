from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import pickle
import numpy as np
import os

# =====================================================
#                 FASTAPI APP SETUP
# =====================================================
app = FastAPI(title="Smart Poultry Prediction API")

# =====================================================
#                 API KEY SECURITY
# =====================================================
API_KEY = ("mysecretkey123")  # Set this in Render Environment
api_key_header = APIKeyHeader(name="X-API-Key")

def check_key(key: str = Security(api_key_header)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API_KEY not set in environment!")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return key

# =====================================================
#                LOAD PICKLED MODELS
# =====================================================

models = {}  
model_dir = "."   # Load .pkl from root directory of repo

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        city_name = file.replace(".pkl", "")
        with open(os.path.join(model_dir, file), "rb") as f:
            models[city_name] = pickle.load(f)

if not models:
    raise Exception("❌ No .pkl models found in repo root!")

print(f"✅ Loaded models: {list(models.keys())}")

# =====================================================
#                        ROUTES
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Poultry Prediction API!",
        "available_models": list(models.keys())
    }

@app.get("/predict")
def predict(
    city: str,
    temp: float,
    humidity: float,
    feed: float,
    weight: float,
    api_key: str = Depends(check_key)
):
    """
    Example URL:
    /predict?city=lahore_rawalpindi_models&temp=30&humidity=60&feed=2.5&weight=1.2

    Header:
        X-API-Key: your_api_key_here
    """

    if city not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{city}' not found. Available: {list(models.keys())}"
        )

    model = models[city]
    X = np.array([[temp, humidity, feed, weight]])
    
    prediction = model.predict(X)

    return {
        "city": city,
        "inputs": {
            "temp": temp,
            "humidity": humidity,
            "feed": feed,
            "weight": weight
        },
        "prediction": prediction.tolist()
    }

