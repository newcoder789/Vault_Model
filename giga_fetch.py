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
import time  # For delay

# Configuration
OPENSEA_API_KEY = os.getenv("open_sea")
TWITTER_BEARER_TOKEN = os.getenv("twitter_bearer_token")
NFT_JSON_FILE = "nfts.json"
CACHE_FILE = "nft_data_cache.json"
CSV_FILE = "nft_risk_data.csv"
MAX_TOTAL_EVENTS = 100000  # Overall cap
EVENTS_PER_PART = 10000  # Target per time part
LIMIT_PER_CALL = 50  # Reduced to default
MAX_CALLS_PER_PART = 100  # Prevent infinite looping

# Set up requests with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# Load cached data with error handling
def load_cache():
    print("Loading cache from", CACHE_FILE)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache_data = json.load(f)
            print(f"Cache loaded with {len(cache_data)} entries")
            return cache_data
        except json.JSONDecodeError as e:
            print(
                f"Invalid JSON in {CACHE_FILE}, ignoring and starting with empty cache: {e}"
            )
            return {}
    print("No cache file found, starting with empty cache")
    return {}


# Save to cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"Cache saved with {len(cache)} entries to {CACHE_FILE}")


# Fetch collection stats from OpenSea including creation date
def fetch_collection_stats(collection_slug):
    print(f"Fetching collection stats for: {collection_slug}")
    url = f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats"
    headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        stats = response.json().get("total", {})
        # Assume creation date is in stats or fallback to nfts.json
        creation_date = stats.get("created_date")
        if not creation_date:
            with open(NFT_JSON_FILE, "r") as f:
                nfts = json.load(f)
                for nft in nfts:
                    if nft["collection"] == collection_slug:
                        creation_date = nft.get("created_date")
                        break
        if not creation_date:
            creation_date = (datetime.now() - timedelta(days=365)).isoformat()
        print(
            f"Stats fetched for {collection_slug}: {stats}, creation_date: {creation_date}"
        )
        return stats, creation_date
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for {collection_slug}: {e}")
        return {}, None


# Fetch sale events from OpenSea with time parts from creation to now
def fetch_sale_events(collection_slug):
    print(f"Fetching sale events for: {collection_slug}")
    url = "https://api.opensea.io/v2/events"
    sales = []

    # Get creation date and total days
    stats, creation_date = fetch_collection_stats(collection_slug)
    if not creation_date:
        creation_date = datetime.now() - timedelta(days=90)
    creation_date = datetime.fromisoformat(creation_date.split("T")[0])
    total_days = (datetime.now() - creation_date).days
    if total_days <= 0:
        total_days = 90
    part_days = max(total_days // 10, 1)

    # Fetch events from creation to now in 10 parts, capped at today
    end_time = datetime.now()
    for i in range(10):
        start_time = end_time - timedelta(days=part_days)
        if start_time < creation_date:
            start_time = creation_date
        params = {
            "collection_slug": collection_slug,
            "event_type": "sale",
            "occurred_after": int(start_time.timestamp()),
            "occurred_before": int(end_time.timestamp()),
            "limit": LIMIT_PER_CALL,
        }
        headers = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
        part_sales = []
        call_count = 0
        print(f"Processing part: {start_time.date()} to {end_time.date()}")
        try:
            while (
                call_count < MAX_CALLS_PER_PART
                and len(part_sales) < EVENTS_PER_PART
                and len(sales) + len(part_sales) < MAX_TOTAL_EVENTS
            ):
                response = session.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                new_sales = data.get("asset_events", [])
                if not new_sales:
                    break
                part_sales.extend(new_sales)
                print(
                    f"Fetched {len(new_sales)} events for {start_time.date()} to {end_time.date()}, total so far: {len(sales) + len(part_sales)}"
                )
                if "next" not in data or not data["next"]:
                    break
                params["cursor"] = data["next"]
                call_count += 1
                time.sleep(1)  # Delay to avoid throttling
            sales.extend(part_sales[:EVENTS_PER_PART])
            print(
                f"Part {start_time.date()} to {end_time.date()} fetched {len(part_sales[:EVENTS_PER_PART])} events"
            )
        except requests.exceptions.RequestException as e:
            print(
                f"Error fetching events for {collection_slug} in {start_time.date()} to {end_time.date()}: {e}, Response: {e.response.text if e.response else 'No response'}"
            )
        end_time = start_time
        if len(sales) >= MAX_TOTAL_EVENTS or start_time <= creation_date:
            break
    print(f"Total sale events fetched for {collection_slug}: {len(sales)}")
    return sales[:MAX_TOTAL_EVENTS]


# Fetch crypto prices from CoinGecko
def fetch_crypto_prices(coin_id):
    print(f"Fetching crypto prices for: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": max((datetime.now() - datetime(2024, 7, 13)).days, 90),
        "interval": "daily",
    }
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        prices = [p[1] for p in response.json().get("prices", [])]
        print(f"Fetched {len(prices)} price points for {coin_id}")
        return prices
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {coin_id} prices: {e}")
        return []


# Fetch Twitter sentiment with recent search
def fetch_twitter_sentiment(collection_name):
    print(f"Fetching Twitter sentiment for: {collection_name}")
    url = "https://api.twitter.com/2/tweets/search/recent"
    query = f"{collection_name} NFT"
    params = {
        "query": query,
        "max_results": 10,
    }
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
        time.sleep(1)
        return mentions, sentiment
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Twitter data for {collection_name}: {e}")
        time.sleep(1)
        return 0, 0


# Compute daily average prices with error handling for payment
def compute_daily_averages(sales):
    print(f"Computing daily averages for {len(sales)} sales")
    if not sales:
        print("No sales data available")
        return pd.Series()
    df = pd.DataFrame(sales)
    df["timestamp"] = pd.to_datetime(df["event_timestamp"], unit="s")
    df["date"] = df["timestamp"].dt.date
    print(f"Unique dates: {df['date'].nunique()}")
    df["price"] = df["payment"].apply(
        lambda x: float(x["quantity"]) / (10 ** x["decimals"])
        if isinstance(x, dict)
        and "quantity" in x
        and "decimals" in x
        and x["token_address"].lower() == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
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
    print("Warning: Insufficient data for volatility, using default 0.5")
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
    with open(NFT_JSON_FILE, "r") as f:
        nfts = json.load(f)
    print(f"Loaded {len(nfts)} NFTs from {NFT_JSON_FILE}")

    collections = list(set(nft["collection"] for nft in nfts))
    print(f"Found {len(collections)} unique collections")
    cache = load_cache()
    dataset = []

    eth_prices = fetch_crypto_prices("ethereum")
    icp_prices = fetch_crypto_prices("internet-computer")

    for collection in collections:
        print(f"Processing collection: {collection}")
        stats = fetch_collection_stats(collection)[0]
        sales = fetch_sale_events(collection)
        daily_avg = compute_daily_averages(sales)
        volatility = compute_volatility(daily_avg)
        total_supply = stats.get("total_supply", 1000)
        trading_volume = stats.get("thirty_day_volume", 0.0)
        floor_price = stats.get("floor_price", 0.0)
        normalized_volume = trading_volume / total_supply if total_supply > 0 else 0.0
        mentions, sentiment = fetch_twitter_sentiment(collection)
        floor_prices = cache.get(collection, {}).get(
            "floor_prices",
            [floor_price] * max((datetime.now() - datetime(2024, 7, 13)).days, 90),
        )
        price_trend = compute_price_trend(floor_prices)
        crypto_correlation = compute_crypto_correlation(daily_avg.values, eth_prices)

        cache.setdefault(collection, {})
        cache[collection]["floor_prices"] = floor_prices[-89:] + [floor_price]
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
        print(
            f"Processed {len([n for n in nfts if n['collection'] == collection])} NFTs for collection {collection}"
        )

    df = pd.DataFrame(dataset)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved {len(df)} records to {CSV_FILE}")
    return df


# Main execution
if __name__ == "__main__":
    df = collect_data()
