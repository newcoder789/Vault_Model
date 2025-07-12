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
OPENSEA_API_KEY = os.getenv("open_sea")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NFT_JSON_FILE = "nfts.json"
CACHE_FILE = "nft_data_cache.json"
CSV_FILE = "nft_risk_data.csv"
DAYS = 30   

# Set up requests with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# Load cached data
def load_cache():
    print("Loading cache from", CACHE_FILE)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)
        print(f"Cache loaded with {len(cache_data)} entries")
        return cache_data
    print("No cache file found, starting with empty cache")
    return {}


# Save to cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"Cache saved with {len(cache)} entries to {CACHE_FILE}")


# Fetch collection stats from OpenSea
def fetch_collection_stats(collection_slug):
    print(f"Fetching collection stats for: {collection_slug}")
    url = f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats"
    headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        stats = response.json().get("total", {})
        print(f"Stats fetched for {collection_slug}: {stats}")
        return stats
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for {collection_slug}: {e}")
        return {}


# Fetch sale events from OpenSea
def fetch_sale_events(collection_slug):
    print(f"Fetching sale events for: {collection_slug}")
    url = "https://api.opensea.io/v2/events"
    params = {
        "collection_slug": collection_slug,
        "event_type": "sale",
        "occurred_after": int((datetime.now() - timedelta(days=DAYS)).timestamp()),
        "limit": 50,
    }
    headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
    sales = []
    try:
        while True:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            sales.extend(data.get("asset_events", []))
            print(f"Fetched {len(data.get('asset_events', []))} events, total so far: {len(sales)}")
            if "next" not in data or not data["next"]:
                break
            params["cursor"] = data["next"]
        print(f"Total sale events fetched for {collection_slug}: {len(sales)}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching events for {collection_slug}: {e}")
    return sales


# Fetch crypto prices from CoinGecko
def fetch_crypto_prices(coin_id):
    print(f"Fetching crypto prices for: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": DAYS, "interval": "daily"}
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        prices = [p[1] for p in response.json().get("prices", [])]
        print(f"Fetched {len(prices)} price points for {coin_id}")
        return prices
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {coin_id} prices: {e}")
        return []


# Fetch Twitter sentiment
def fetch_twitter_sentiment(collection_name):
    print(f"Fetching Twitter sentiment for: {collection_name}")
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
        print(f"Twitter mentions: {mentions}, sentiment: {sentiment:.4f}")
        return mentions, sentiment
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Twitter data for {collection_name}: {e}")
        return 0, 0


# Compute daily average prices
def compute_daily_averages(sales):
    print(f"Computing daily averages for {len(sales)} sales")
    if not sales:
        print("No sales data available")
        return pd.Series()
    df = pd.DataFrame(sales)
    df["timestamp"] = pd.to_datetime(df["event_timestamp"], unit="s")
    df["date"] = df["timestamp"].dt.date
    df["price"] = df["payment"].apply(
        lambda x: float(x["quantity"]) / (10 ** x["decimals"])
        if x["token_address"].lower() == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        else 0
    )
    daily_avg = df.groupby("date")["price"].mean()
    print(f"Computed daily averages for {len(daily_avg)} days")
    return daily_avg


# Compute volatility
def compute_volatility(daily_avg):
    print("Computing volatility")
    if len(daily_avg) > 1 and np.mean(daily_avg) > 0:
        volatility = np.std(daily_avg) / np.mean(daily_avg)
        print(f"Volatility computed: {volatility:.4f}")
        return volatility
    print("Insufficient data for volatility, returning default 0.5")
    return 0.5


# Compute price trend
def compute_price_trend(prices):
    print("Computing price trend")
    if len(prices) < 2:
        print("Not enough data points for price trend")
        return 0
    x = np.arange(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    print(f"Price trend slope: {slope:.6f}")
    return slope


# Compute crypto correlation
def compute_crypto_correlation(nft_prices, crypto_prices):
    print("Computing crypto correlation")
    if len(nft_prices) < 2 or len(crypto_prices) < 2:
        print("Not enough data points for correlation")
        return 0
    min_len = min(len(nft_prices), len(crypto_prices))
    corr = np.corrcoef(nft_prices[:min_len], crypto_prices[:min_len])[0, 1]
    print(f"Crypto correlation: {corr:.4f}")
    return corr


# Compute risk score
def compute_risk_score(features):
    print("Computing risk score")
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
    risk = min(risk, 100.0)
    print(f"Risk score computed: {risk:.2f}")
    return risk


# Main data collection and processing
def collect_data():
    print("Starting data collection")
    with open("nfts2.json", "r") as f:
        nfts = json.load(f)
    print(f"Loaded {len(nfts)} NFTs from nfts.json")

    collections = list(set(nft["collection"] for nft in nfts))
    print(f"Found {len(collections)} unique collections")
    cache = load_cache()
    dataset = []

    eth_prices = fetch_crypto_prices("ethereum")
    icp_prices = fetch_crypto_prices("internet-computer")

    for collection in collections:
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

        for nft in [n for n in nfts if n["collection"] == collection]:
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
                    "token_id": nft["identifier"],
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
        print(f"Processed {len([n for n in nfts if n['collection'] == collection])} NFTs for collection {collection}")

    df = pd.DataFrame(dataset)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved {len(df)} records to {CSV_FILE}")
    return df


# Main execution
if __name__ == "__main__":
    df = collect_data()
