from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import re
import os
import json
import joblib
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ic_agent import Agent
from ic_agent.identity import AnonymousIdentity
from ic_utils import Principal
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration 
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY", "24e211c34b284ce4bea594c062ba11bf")
CACHE_FILE = "nft_data_cache.json"

# ml model
try:
    model = joblib.load("nft_risk_model.pkl")
    logger.info("NFT risk model loaded successfully.")
except Exception as e:
    model = None
    logger.error(f"Could not load nft_risk_model.pkl: {e}. NFT risk scoring will fail.")

app = FastAPI()


class ScoreRequest(BaseModel):
    principal: str
    nft_contract: str
    nft_id: str


#
class ScoreResponse(BaseModel):
    credit_score: int
    nft_risk_score: int
    composite_score: int


CREDIT_WEIGHTS = {
    "wallet_balance": 0.3,
    "nft_portfolio": 0.25,
    "web3_reputation": 0.2,
    "traditional_credit": 0.15,
    "loan_history": 0.1,
}
COMPOSITE_WEIGHTS = {"credit": 0.4, "nft": 0.6}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# Validate ICP principal format
def is_valid_principal(principal: str) -> bool:
    return bool(re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", principal))


# Fetch ICP balance using ic-agent-python
async def get_icp_balance(principal: str) -> float:
    try:
        agent = Agent("https://ic0.app", identity=AnonymousIdentity())
        ledger_canister_id = "ryjl3-tyaaa-aaaaa-aaaba-cai"
        principal_obj = Principal.from_text(principal)
        response = await agent.query(
            ledger_canister_id,
            "icrc1_balance_of",
            {"owner": {"owner": principal_obj}, "subaccount": None},
        )
        balance = int.from_bytes(response, "big") / 1_000_000_000  # Convert e8s to ICP
        normalized_balance = min(int(balance / 100 * 100), 100)  # Normalize to 0-100
        return normalized_balance
    except Exception as e:
        logger.error(f"Failed to fetch ICP balance for {principal}: {str(e)}")
        return 50  # Fallback score


# Fetch collection stats from OpenSea API for Ethereum NFTs
def fetch_collection_stats(collection_slug: str) -> dict:
    url = f"https://api.opensea.io/v2/collections/{collection_slug}/stats"
    headers = {
        "X-API-KEY": OPENSEA_API_KEY,
        "accept": "application/json",
        "User-Agent": "VaulticScoringAPI/1.0",
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("total", {})
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        logger.error(
            f"HTTP error fetching OpenSea data for {collection_slug}: {http_err}"
        )
        if status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_slug}' not found"
            )
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Upstream API error")
    except Exception as e:
        logger.error(
            f"Failed to fetch collection stats for {collection_slug}: {str(e)}"
        )
        return {
            "floor_price": 0.0,
            "thirty_day_volume": 0.0,
            "total_supply": 1000,
        }


# Fetch NFT data from ICP canister (placeholder)
async def fetch_icp_nft_data(nft_contract: str, nft_id: str) -> dict:
    try:
        agent = Agent("https://ic0.app", identity=AnonymousIdentity())
        response = await agent.query(
            nft_contract, "icrc7_metadata", {"token_id": nft_id}
        )
        metadata = json.loads(response)  # Assuming JSON response
        total_supply_response = await agent.query(nft_contract, "icrc7_total_supply")
        total_supply = int.from_bytes(total_supply_response, "big")
        return {
            "total_supply": total_supply,
            "floor_price": 0.0,  # Placeholder until marketplace API available
            "thirty_day_volume": 0.0,  # Placeholder
            "rarity_score": 0.5,  # Placeholder
        }
    except Exception as e:
        logger.error(
            f"Failed to fetch ICP NFT data for {nft_contract}/{nft_id}: {str(e)}"
        )
        return {
            "total_supply": 1000,
            "floor_price": 0.0,
            "thirty_day_volume": 0.0,
            "rarity_score": 0.5,
        }


# Fetch all NFTs owned by a principaL --- TESTING NOT WORKING 
def fetch_user_nfts(principal: str) -> float:
    url = f"https://entrepot.app/api/nfts?principal={principal}"  # Hypothetical
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nfts = response.json().get("nfts", [])
        total_value = sum(nft.get("floor_price", 0.0) for nft in nfts)
        normalized_value = min(int(total_value / 1000 * 100), 100)  # Normalize to 0-100
        return normalized_value
    except Exception as e:
        logger.error(f"Failed to fetch user NFTs for {principal}: {str(e)}")
        return 50  # Fallback score


# Fetch credit data
async def fetch_credit_data(principal: str) -> dict:
    if not is_valid_principal(principal):
        logger.error(f"Invalid principal: {principal}")
        raise HTTPException(status_code=400, detail="Invalid ICP principal format")
    try:
        wallet_balance = await get_icp_balance(principal)
        nft_portfolio = fetch_user_nfts(principal)  # Hypothetical
        return {
            "wallet_balance": wallet_balance,
            "nft_portfolio": nft_portfolio,
            "web3_reputation": 50,  # Placeholder until API available
            "traditional_credit": None,
            "loan_history": 50,  # Placeholder for new users
        }
    except Exception as e:
        logger.error(f"Credit data fetch failed for {principal}: {str(e)}")
        return {
            "wallet_balance": 50,
            "nft_portfolio": 50,
            "web3_reputation": 50,
            "traditional_credit": None,
            "loan_history": 50,
        }


# Load and save cache
def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# Compute features for ML model
def compute_features_for_model(data: dict, collection_slug: str, nft_id: str) -> dict:
    cache = load_cache()
    collection_size = data.get("total_supply", 1000)
    floor_price = data.get("floor_price", 0.0)
    volume = data.get("thirty_day_volume", 0.0)
    floor_prices = cache.get(collection_slug, {}).get(
        "floor_prices", [floor_price] * 30
    )
    if floor_prices and np.mean(floor_prices) > 0:
        volatility = np.std(floor_prices) / np.mean(floor_prices)
    else:
        volatility = 0.5
    norm_volume = volume / collection_size if collection_size > 0 else 0.0
    rarity_score = cache.get(collection_slug, {}).get("rarity_score", 0.5)
    cache.setdefault(collection_slug, {})
    cache[collection_slug]["floor_prices"] = (floor_prices + [floor_price])[-30:]
    cache[collection_slug]["rarity_score"] = rarity_score
    save_cache(cache)
    return {
        "volatility": volatility,
        "norm_volume": norm_volume,
        "collection_size": collection_size,
        "rarity_score": rarity_score,
    }


# Scoring functions
def compute_credit_score(data: dict) -> int:
    wallet = data["wallet_balance"]
    nft = data["nft_portfolio"]
    web3 = data["web3_reputation"]
    traditional = (
        data["traditional_credit"] if data["traditional_credit"] is not None else 0
    )
    loan = data["loan_history"]
    score = (
        CREDIT_WEIGHTS["wallet_balance"] * wallet
        + CREDIT_WEIGHTS["nft_portfolio"] * nft
        + CREDIT_WEIGHTS["web3_reputation"] * web3
        + CREDIT_WEIGHTS["traditional_credit"] * traditional
        + CREDIT_WEIGHTS["loan_history"] * loan
    )
    return min(int(score), 100)


def predict_nft_risk_score(features: dict) -> int:
    if model is None:
        raise RuntimeError("NFT risk model is not loaded.")
    feature_array = np.array(
        [
            [
                features["volatility"],
                features["norm_volume"],
                features["collection_size"],
                features["rarity_score"],
            ]
        ]
    )
    predicted_risk = model.predict(feature_array)[0]
    return min(max(100 - int(predicted_risk), 0), 100)


def compute_composite_score(credit_score: int, nft_risk_score: int) -> int:
    score = (
        COMPOSITE_WEIGHTS["credit"] * credit_score
        + COMPOSITE_WEIGHTS["nft"] * nft_risk_score
    )
    return min(int(score), 100)


# API endpoint
@app.get("/get_scores", response_model=ScoreResponse)
async def get_scores(principal: str, nft_contract: str, nft_id: str):
    if not is_valid_principal(principal):
        raise HTTPException(status_code=400, detail="Invalid ICP principal format")
    try:
        credit_data = await fetch_credit_data(principal)
        # Assume nft_contract is an OpenSea collection slug for Ethereum NFTs
        # For ICP NFTs, replace with canister query logic
        nft_collection_data = (
            fetch_collection_stats(nft_contract)
            if not nft_contract.startswith("aaaaa-")
            else await fetch_icp_nft_data(nft_contract, nft_id)
        )
        nft_features = compute_features_for_model(
            nft_collection_data, nft_contract, nft_id
        )
        credit_score = compute_credit_score(credit_data)
        nft_risk_score = predict_nft_risk_score(nft_features)
        composite_score = compute_composite_score(credit_score, nft_risk_score)
        return ScoreResponse(
            credit_score=credit_score,
            nft_risk_score=nft_risk_score,
            composite_score=composite_score,
        )
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"NFT collection '{nft_contract}' not found"
            )
        logger.error(f"HTTP error fetching data for {nft_contract}: {http_err}")
        raise HTTPException(
            status_code=502,
            detail=f"Bad Gateway: Upstream API error for {nft_contract}",
        )
    except RuntimeError as runtime_err:
        logger.error(f"Runtime error during scoring for {principal}: {runtime_err}")
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {runtime_err}"
        )
    except Exception as e:
        logger.error(f"Scoring failed for principal {principal}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


# Run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
