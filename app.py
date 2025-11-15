from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import pickle
import numpy as np
import os

# ========== FASTAPI SETUP ==========
app = FastAPI(title="Smart Poultry Prediction API")

# ========== API KEY SETUP ==========
API_KEY = os.getenv("API_KEY") # you can change this if needed
api_key_header = APIKeyHeader(name="X-API-Key")

def check_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return key

# ========== LOAD ALL PICKLED MODELS ==========
model_dir = "pickled_models"
models = {}

if not os.path.exists(model_dir):
    raise Exception(f"Folder '{model_dir}' not found. Please create it and add .pkl models.")

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        city_name = file.replace(".pkl", "")
        with open(os.path.join(model_dir, file), "rb") as f:
            models[city_name] = pickle.load(f)

if not models:
    raise Exception("No models found in pickled_models folder!")

print(f"✅ Loaded models: {list(models.keys())}")

# ========== ROUTES ==========

@app.get("/")
def home():
    """Root endpoint"""
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
    Example:
    http://127.0.0.1:8000/predict?city=Faisalabad_close_mlp&temp=30&humidity=60&feed=2.5&weight=1.2
    Header: X-API-Key = mysecretkey123
    """

    # Check if city exists
    if city not in models:
        raise HTTPException(status_code=404, detail=f"Model for '{city}' not found.")

    model = models[city]
    X = np.array([[temp, humidity, feed, weight]])
    prediction = model.predict(X)
    
    return {
        "city": city,
        "inputs": {"temp": temp, "humidity": humidity, "feed": feed, "weight": weight},
        "prediction": prediction.tolist()
    }
