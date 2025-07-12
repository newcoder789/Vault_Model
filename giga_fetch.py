import requests
import pandas as pd
import numpy as np
from scipy.stats import linregress
from textblob import TextBlob
from datetime import datetime, timedelta
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY", "24e211c34b284ce4bea594c062ba11bf")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NFT_JSON_FILE = "nfts.json"
CACHE_FILE = "nft_data_cache.json"
CSV_FILE = "./output/nft_risk_data.csv"  
DAYS = 30
MAX_EVENTS_PER_COLLECTION = 500

# Ensure output directory exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Set up requests with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


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


# Fetch collection stats from OpenSea
def fetch_collection_stats(collection_slug):
    url = f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats"
    headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("total", {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for {collection_slug}: {e}")
        return {}


# Fetch sale events from OpenSea
def fetch_sale_events(collection_slug):
    url = "https://api.opensea.io/v2/events"
    params = {
        "collection_slug": collection_slug,
        "event_type": "sale",
        "occurred_after": int((datetime.now() - timedelta(days=DAYS)).timestamp()),
        "limit": 50,
    }
    headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
    sales = []
    total_fetched = 0
    try:
        while total_fetched < MAX_EVENTS_PER_COLLECTION:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            new_sales = data.get("asset_events", [])
            sales.extend(new_sales)
            total_fetched += len(new_sales)
            print(
                f"Fetched {len(new_sales)} events for {collection_slug}, total: {total_fetched}"
            )
            if "next" not in data or not data["next"] or not new_sales:
                break
            params["cursor"] = data["next"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching events for {collection_slug}: {e}")
    return sales[:MAX_EVENTS_PER_COLLECTION]


# Fetch crypto prices from CoinGecko
def fetch_crypto_prices(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": DAYS, "interval": "daily"}
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return [p[1] for p in response.json().get("prices", [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {coin_id} prices: {e}")
        return [0] * DAYS


# Fetch Twitter sentiment
def fetch_twitter_sentiment(collection_name):
    url = "https://api.twitter.com/2/tweets/search/recent"
    query = f"{collection_name} NFT"
    params = {"query": query, "max_results": 10}
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    try:
        response = session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        tweets = response.json().get("data", [])
        mentions = len(tweets)
        sentiment = (
            np.mean([TextBlob(tweet["text"]).sentiment.polarity for tweet in tweets])
            if tweets
            else 0
        )
        return mentions, sentiment
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Twitter data for {collection_name}: {e}")
        return 0, 0


# Compute daily average prices
def compute_daily_averages(sales):
    if not sales:
        return pd.Series()
    df = pd.DataFrame(sales)
    df["timestamp"] = pd.to_datetime(df["event_timestamp"], unit="s")
    df["date"] = df["timestamp"].dt.date
    df["price"] = df["payment"].apply(
        lambda x: float(x["quantity"]) / (10 ** x["decimals"])
        if x.get("token_address", "").lower()
        == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        else 0
    )
    return df.groupby("date")["price"].mean()


# Compute volatility
def compute_volatility(daily_avg):
    if len(daily_avg) > 1 and np.mean(daily_avg) > 0:
        return np.std(daily_avg) / np.mean(daily_avg)
    return 0.5


# Compute price trend
def compute_price_trend(prices):
    if len(prices) < 2:
        return 0
    x = np.arange(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    return slope


# Compute crypto correlation
def compute_crypto_correlation(nft_prices, crypto_prices):
    if len(nft_prices) < 2 or len(crypto_prices) < 2:
        return 0
    min_len = min(len(nft_prices), len(crypto_prices))
    return np.corrcoef(nft_prices[:min_len], crypto_prices[:min_len])[0, 1]


# Compute risk score
def compute_risk_score(features):
    scaled_volatility = features["volatility"] * 100
    scaled_norm_volume = 1 / (features["normalized_volume"] + 1e-6) * 10
    scaled_collection_size = features["collection_size"] / 10000
    scaled_rarity = 1 - features["rarity_score"]
    scaled_price_trend = (features["price_trend"] + 1) * 50
    scaled_crypto_correlation = features["crypto_correlation"] * 100
    scaled_sentiment = (1 - features["sentiment_score"]) * 50
    risk = (
        0.30 * scaled_volatility
        + 0.25 * scaled_norm_volume
        + 0.15 * scaled_collection_size
        + 0.15 * scaled_rarity
        + 0.10 * scaled_price_trend
        + 0.05 * scaled_crypto_correlation
        + 0.05 * scaled_sentiment
    )
    return min(risk, 100.0)


# Main data collection and processing
def collect_data():
    if not os.path.exists(NFT_JSON_FILE) or os.path.getsize(NFT_JSON_FILE) == 0:
        print(f"Error: '{NFT_JSON_FILE}' is missing or empty.")
        print("Please ensure it contains valid NFT data in JSON format.")
        return pd.DataFrame()

    try:
        with open(NFT_JSON_FILE, "r") as f:
            nfts = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{NFT_JSON_FILE}': {e}")
        return pd.DataFrame()
 
    collections = list(set(nft["collection"] for nft in nfts if nft.get("collection")))
    cache = load_cache()
    dataset = []

    eth_prices = fetch_crypto_prices("ethereum")
    icp_prices = fetch_crypto_prices("internet-computer")

    for collection in collections:
        if not collection:
            continue
        print(f"Processing collection: {collection}")
        stats = fetch_collection_stats(collection)
        sales = fetch_sale_events(collection)
        daily_avg = compute_daily_averages(sales)
        volatility = compute_volatility(daily_avg)
        total_supply = stats.get("total_supply", 1000)
        trading_volume = stats.get("thirty_day_volume", 0.0)
        floor_price = stats.get("floor_price", 0.0)
        normalized_volume = trading_volume / total_supply if total_supply > 0 else 0.0
        mentions, sentiment = fetch_twitter_sentiment(collection)
        floor_prices = cache.get(collection, {}).get(
            "floor_prices", [floor_price] * DAYS
        )
        price_trend = compute_price_trend(floor_prices)
        crypto_correlation = compute_crypto_correlation(daily_avg.values, eth_prices)

        cache.setdefault(collection, {})
        cache[collection]["floor_prices"] = floor_prices[-29:] + [floor_price]
        save_cache(cache)

        for nft in [n for n in nfts if n.get("collection") == collection]:
            rank = nft.get("rarity", {}).get("rank", total_supply // 2)
            rarity_score = (
                (total_supply - rank) / total_supply if total_supply > 0 else 0.5
            )
            features = {
                "volatility": volatility,
                "normalized_volume": normalized_volume,
                "collection_size": total_supply,
                "rarity_score": rarity_score,
                "price_trend": price_trend,
                "crypto_correlation": crypto_correlation,
                "sentiment_score": sentiment,
            }
            risk_score = compute_risk_score(features)
            dataset.append(
                {
                    "collection_id": collection,
                    "token_id": nft.get("identifier", ""),
                    "floor_price": floor_price,
                    "trading_volume": trading_volume,
                    "total_supply": total_supply,
                    "rarity_rank": rank,
                    "sale_prices": json.dumps(list(daily_avg.values)),
                    "timestamps": json.dumps([str(d) for d in daily_avg.index]),
                    "eth_prices": json.dumps(eth_prices),
                    "icp_prices": json.dumps(icp_prices),
                    "twitter_mentions": mentions,
                    "volatility": volatility,
                    "normalized_volume": normalized_volume,
                    "collection_size": total_supply,
                    "rarity_score": rarity_score,
                    "price_trend": price_trend,
                    "crypto_correlation": crypto_correlation,
                    "sentiment_score": sentiment,
                    "risk_score": risk_score,
                }
            )

    df = pd.DataFrame(dataset)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved {len(df)} records to {CSV_FILE}")
    return df


# Main execution
if __name__ == "__main__":
    try:
        df = collect_data()
        if df.empty:
            print("No data was processed. Exiting.")
    except Exception as e:
        print(f"Script failed: {e}")
