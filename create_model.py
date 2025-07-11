import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import time
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY", "open_sea")
COLLECTIONS = ["bored-ape-yacht-club", "cryptopunks", "azuki"]
DATA_DAYS = 30
CACHE_FILE = "nft_data_cache.json"

# Set up requests with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Fetch data from OpenSea API (v2)
def fetch_collection_data(collection_slug):
    url = f"https://api.opensea.io/v2/collections/{collection_slug}/stats"

    headers = {
        "X-API-KEY": OPENSEA_API_KEY,
        "accept": "application/json",
        "User-Agent": "NFTLoanRiskModel/1.0"
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        print("Fetching data for:", response.json())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {collection_slug}: {e}")
        return None

# Load cached data
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save to cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Compute features
def compute_features(data, collection_slug):
    cache = load_cache()
    collection_size = data.get("total_supply", 1000)  # Default if missing
    floor_price = data.get("floor_price", 0.0)
    volume = data.get("thirty_day_volume", 0.0)
    
    # Placeholder for historical floor prices (requires separate API or database)
    floor_prices = cache.get(collection_slug, {}).get("floor_prices", [floor_price] * 30)
    volatility = np.std(floor_prices) / np.mean(floor_prices) if floor_prices and np.mean(floor_prices) > 0 else 0.5
    norm_volume = volume / collection_size if collection_size > 0 else 0.0
    rarity_score = cache.get(collection_slug, {}).get("rarity_score", 0.5)  # Placeholder
    
    # Update cache
    cache.setdefault(collection_slug, {})
    cache[collection_slug]["floor_prices"] = floor_prices[-30:] + [floor_price]
    cache[collection_slug]["rarity_score"] = rarity_score
    save_cache(cache)
    
    return {
        "volatility": volatility,
        "norm_volume": norm_volume,
        "collection_size": collection_size,
        "rarity_score": rarity_score
    }

# Rule-based risk score
def rule_based_risk_score(features):
    return (0.4 * features["volatility"] + 
            0.3 * (1.0 / max(features["norm_volume"], 1e-6)) + 
            0.2 * (1.0 / max(features["collection_size"], 1)) + 
            0.1 * features["rarity_score"]) * 100

# Collect and preprocess data
def collect_data():
    dataset = []
    for collection in COLLECTIONS:
        data = fetch_collection_data(collection)
        if not data:
            print(f"Using cached data for {collection}")
            cache = load_cache()
            data = cache.get(collection, {})
            if not data:
                continue
        features = compute_features(data, collection)
        risk_score = rule_based_risk_score(features)
        dataset.append({
            "collection": collection,
            "volatility": features["volatility"],
            "norm_volume": features["norm_volume"],
            "collection_size": features["collection_size"],
            "rarity_score": features["rarity_score"],
            "risk_score": min(risk_score, 100.0)  # Cap at 100
        })
    return pd.DataFrame(dataset)

# Train ML model
def train_model(df):
    if df.empty:
        raise ValueError("No data available for training")
    
    X = df[["volatility", "norm_volume", "collection_size", "rarity_score"]]
    y = df["risk_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, R²: {r2:.2f}")
    
    joblib.dump(best_model, "nft_risk_model.pkl")
    return best_model

# Inference function
def predict_risk_score(features):
    model = joblib.load("nft_risk_model.pkl")
    X = np.array([[features["volatility"], features["norm_volume"], 
                   features["collection_size"], features["rarity_score"]]])
    return model.predict(X)[0]

# Main execution
if __name__ == "__main__":
    df = collect_data()
    if not df.empty:
        model = train_model(df)
        sample_features = {
            "volatility": 0.2,
            "norm_volume": 0.01,
            "collection_size": 10000,
            "rarity_score": 0.8
        }
        risk_score = predict_risk_score(sample_features)
        print(f"Predicted Risk Score: {risk_score:.2f}")
    else:
        print("Failed to collect data. Check API key, endpoints, or network.")