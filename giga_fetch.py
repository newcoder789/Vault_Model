import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import sys
import json
from datetime import datetime
import time

# 1. API Setup
OPENSEA_API_KEY = os.getenv("opensea_api_key")
ALCHEMY_API_KEY = os.getenv("alchemy_key", "6q_qZNutwscvksSLiK7dYq6NbQ7ddFSd")
TWITTER_BEARER_TOKEN = os.getenv("twitter_bearer_token")
if not OPENSEA_API_KEY:
    print(
        "Error: OPENSEA_API_KEY environment variable not set. Sign up at opensea.io for an API key."
    )
    sys.exit(1)
if not ALCHEMY_API_KEY:
    print("Error: ALCHEMY_API_KEY environment variable not set.")
    sys.exit(1)
if not TWITTER_BEARER_TOKEN:
    print("Error: TWITTER_BEARER_TOKEN environment variable not set.")
    sys.exit(1)

BASE_URL_OPENSEA = "https://api.opensea.io/api/v2"
HEADERS_OPENSEA = {"X-API-KEY": OPENSEA_API_KEY, "accept": "application/json"}
ALCHEMY_BASE_URL = f"https://eth-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"
TWITTER_HEADERS = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

# 2. Hardcoded Collection Slugs (Temporary Workaround)
all_slugs = [
    "boredapeyachtclub",
    "mutant-ape-yacht-club",
    "cryptopunks",
]  # Add more as needed
print(f"Using hardcoded slugs: {all_slugs}")


# 3. Fetch NFTs from Collections (OpenSea)
def fetch_collection_nfts(slug, limit=200, offset=0, max_retries=3):
    url = f"{BASE_URL_OPENSEA}/collection/{slug}/nfts"
    params = {"limit": limit, "offset": offset}
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=HEADERS_OPENSEA, params=params, timeout=10
            )
            if response.status_code == 200:
                return response.json().get("nfts", [])
            else:
                print(
                    f"Attempt {attempt + 1} failed for {slug}: {response.status_code} - {response.text}"
                )
                if response.status_code == 429:
                    time.sleep(2**attempt)
                else:
                    break
        except requests.RequestException as e:
            print(
                f"Network error for {slug}: {e}. Attempt {attempt + 1} of {max_retries}"
            )
            time.sleep(2**attempt)
    return []


all_nfts_opensea = []
for slug in all_slugs[:5]:  # Limit to 5 for now
    offset = 0
    while True:
        nfts = fetch_collection_nfts(slug, offset=offset)
        if not nfts:
            break
        all_nfts_opensea.extend(nfts)
        offset += 200
        if len(nfts) < 200:
            break

if not all_nfts_opensea:
    print("No NFTs fetched from OpenSea. Check API limits or data.")
    sys.exit(1)

# 4. Fetch NFTs from Alchemy for Owner
OWNER_ADDRESS = "0xa858ddc0445d8131dac4d1de01f834ffcba52ef1"


def fetch_alchemy_nfts(owner, max_retries=3):
    url = f"{ALCHEMY_BASE_URL}/getNFTsForOwner?owner={owner}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("ownedNfts", [])
            else:
                print(
                    f"Attempt {attempt + 1} failed for Alchemy: {response.status_code} - {response.text}"
                )
                time.sleep(2**attempt)
                break
        except requests.RequestException as e:
            print(
                f"Network error for Alchemy: {e}. Attempt {attempt + 1} of {max_retries}"
            )
            time.sleep(2**attempt)
    return []


all_nfts_alchemy = fetch_alchemy_nfts(OWNER_ADDRESS)


# 5. Fetch Twitter Data
def fetch_twitter_data(max_retries=3):
    url = "https://api.twitter.com/2/tweets/search/recent?query=from:BoredApeYC&max_results=100"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=TWITTER_HEADERS, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"Attempt {attempt + 1} failed for Twitter: {response.status_code} - {response.text}"
                )
                if response.status_code == 429:
                    time.sleep(2**attempt)
                else:
                    break
        except requests.RequestException as e:
            print(
                f"Network error for Twitter: {e}. Attempt {attempt + 1} of {max_retries}"
            )
            time.sleep(2**attempt)
    return {}


twitter_data = fetch_twitter_data()
sentiment_score = 0
tweet_count = len(twitter_data.get("data", []))
if tweet_count > 0:
    for tweet in twitter_data.get("data", []):
        sentiment_score += 0.5  # Dummy positive sentiment
    sentiment_score /= tweet_count
retweets_sum = sum(
    tweet.get("public_metrics", {}).get("retweet_count", 0)
    for tweet in twitter_data.get("data", [])
)
likes_sum = sum(
    tweet.get("public_metrics", {}).get("like_count", 0)
    for tweet in twitter_data.get("data", [])
)
avg_retweets = retweets_sum / tweet_count if tweet_count else 0
avg_likes = likes_sum / tweet_count if tweet_count else 0


# 6. Fetch Collection Metadata for Spam Filtering
def fetch_collection_metadata(slug, max_retries=3):
    url = f"{BASE_URL_OPENSEA}/collection/{slug}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS_OPENSEA, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"Attempt {attempt + 1} failed for {slug} metadata: {response.status_code} - {response.text}"
                )
                if response.status_code == 429:
                    time.sleep(2**attempt)
                else:
                    break
        except requests.RequestException as e:
            print(
                f"Network error for {slug} metadata: {e}. Attempt {attempt + 1} of {max_retries}"
            )
            time.sleep(2**attempt)
    return {}


# 7. Process and Combine Data
data = {
    "owner_address": [
        nft.get("owner", {}).get("address", "") for nft in all_nfts_opensea
    ]
    + [nft.get("owner_address", "") for nft in all_nfts_alchemy],
    "contract_address": [
        nft.get("contract", {}).get("address", "") for nft in all_nfts_opensea
    ]
    + [nft["contract"]["address"] for nft in all_nfts_alchemy],
    "token_id": [nft.get("token_id", "") for nft in all_nfts_opensea]
    + [nft["tokenId"] for nft in all_nfts_alchemy],
    "token_type": [nft.get("token_type", "ERC721") for nft in all_nfts_opensea]
    + [nft["tokenType"] for nft in all_nfts_alchemy],
    "balance": [nft.get("balance", 1) for nft in all_nfts_opensea]
    + [nft["balance"] for nft in all_nfts_alchemy],
    "collection_name": [
        nft.get("collection", {}).get("name", "") for nft in all_nfts_opensea
    ]
    + [nft["collection"]["name"] for nft in all_nfts_alchemy],
    "collection_slug": [
        nft.get("collection", {}).get("slug", "") for nft in all_nfts_opensea
    ]
    + [nft["collection"]["slug"] for nft in all_nfts_alchemy],
    "is_spam": [False] * (len(all_nfts_opensea) + len(all_nfts_alchemy)),
    "floor_price": [
        nft.get("collection", {}).get("stats", {}).get("floor_price", 0)
        for nft in all_nfts_opensea
    ]
    + [nft["contract"]["openSeaMetadata"]["floorPrice"] for nft in all_nfts_alchemy],
    "image_url": [nft.get("image_url", "") for nft in all_nfts_opensea]
    + [nft["contract"]["openSeaMetadata"]["imageUrl"] for nft in all_nfts_alchemy],
    "description": [nft.get("description", "") for nft in all_nfts_opensea]
    + [nft["contract"]["openSeaMetadata"]["description"] for nft in all_nfts_alchemy],
    "last_ingested_at": [nft.get("last_ingested_at", "") for nft in all_nfts_opensea]
    + [
        nft["contract"]["openSeaMetadata"]["lastIngestedAt"] for nft in all_nfts_alchemy
    ],
    "total_supply": [
        nft.get("collection", {}).get("stats", {}).get("total_supply", 0)
        for nft in all_nfts_opensea
    ]
    + [nft["contract"]["totalSupply"] for nft in all_nfts_alchemy],
    "twitter_sentiment_score": [sentiment_score]
    * (len(all_nfts_opensea) + len(all_nfts_alchemy)),
    "tweet_count_last_24h": [tweet_count]
    * (len(all_nfts_opensea) + len(all_nfts_alchemy)),
    "avg_retweets_last_24h": [avg_retweets]
    * (len(all_nfts_opensea) + len(all_nfts_alchemy)),
    "avg_likes_last_24h": [avg_likes] * (len(all_nfts_opensea) + len(all_nfts_alchemy)),
}

df = pd.DataFrame(data)


# 8. Enhanced Spam Detection
def is_spam(nft_data, collection_meta):
    metadata = nft_data.get("description", "").lower()
    image = nft_data.get("image_url", "")
    floor_price = nft_data.get("floor_price", 0)
    safelist_status = collection_meta.get("safelist_status", "not_requested")
    volume = collection_meta.get("stats", {}).get("total_volume", 0)
    # Use Alchemy isSpam if available
    is_spam_alchemy = (
        nft_data.get("contract", {}).get("isSpam", False)
        if isinstance(nft_data, dict) and "contract" in nft_data
        else False
    )
    return (
        is_spam_alchemy  # Alchemy spam flag
        or not metadata  # Empty description
        or "http" not in image  # No valid image URL
        or "spam" in metadata  # Keyword check
        or floor_price == 0  # Zero floor price
        or safelist_status == "not_requested"  # Not safelisted
        or volume < 10  # Low trading volume
    )


# Apply spam filtering with collection metadata
df["collection_meta"] = [
    fetch_collection_metadata(nft["collection"]["slug"]) if "collection" in nft else {}
    for nft in all_nfts_opensea + all_nfts_alchemy
]
df["is_spam"] = [
    is_spam(nft, meta)
    for nft, meta in zip(all_nfts_opensea + all_nfts_alchemy, df["collection_meta"])
]
df = df[df["is_spam"] == False].reset_index(drop=True)

# Clean up temporary column
df = df.drop(columns=["collection_meta"])

# Save raw and cleaned data
df.to_csv("nft_data_raw.csv", index=False)
df.to_csv("nft_data_cleaned.csv", index=False)

# 9. ML Model (Optional)
if not df.empty and len(df) > 10:
    # Preprocess numeric columns
    numeric_features = [
        "floor_price",
        "balance",
        "total_supply",
        "twitter_sentiment_score",
        "tweet_count_last_24h",
        "avg_retweets_last_24h",
        "avg_likes_last_24h",
    ]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[numeric_features].copy()

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    categorical_cols = ["token_type", "collection_slug"]
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = ""
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_cols, columns=encoder.get_feature_names_out(categorical_cols)
    )
    X = pd.concat([X, encoded_df], axis=1)

    # Train-test split and model
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["is_spam"], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
else:
    print("Not enough non-spam data to train model.")

print(
    f"Collected {len(all_nfts_opensea) + len(all_nfts_alchemy)} NFTs, kept {len(df)} after spam filtering."
)
