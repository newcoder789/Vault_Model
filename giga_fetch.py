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
from dotenv import load_dotenv

# 1. API Setup
open_sea_key = os.getenv("OPENSEA_API_KEY", "24e211c34b284ce4bea594c062ba11bf")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "6q_qZNutwscvksSLiK7dYq6NbQ7ddFSd")

BASE_URL_OPENSEA = "https://api.opensea.io/api/v2"
HEADERS_OPENSEA = {"X-API-KEY": open_sea_key, "accept": "application/json"}
ALCHEMY_BASE_URL = f"https://eth-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"


# 2. Fetch slugs (unchanged but not used here since we’re using preloaded)
def fetch_collections():
    url = "https://api.opensea.io/api/v2/collections?order_by=market_cap"
    headers = {"accept": "application/json", "x-api-key": open_sea_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch collections: {response.status_code} - {response.text}")
        return {"collections": []}
    return response.json()


# 3. Fetch NFTs from Collections (OpenSea) - unused here but kept
def fetch_collection_nfts(slug, max_retries=3):
    url = f"{BASE_URL_OPENSEA}/collection/{slug}/nfts"
    params = {"limit": 25}
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=HEADERS_OPENSEA, params=params, timeout=10
            )
            print(
                f"Response for {slug}: {response.status_code} - {response.text[:200]}... "
            )
            if response.status_code == 200:
                data = response.json()
                nfts = data.get("nfts", [])
                if not isinstance(nfts, list):
                    print(f"Unexpected data format for {slug}: {data}")
                    return []
                return nfts
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


# 8. Enhanced Spam Detection
def is_spam(nft_data, collection_meta):
    metadata = nft_data.get("description", "").lower()
    image = nft_data.get("image_url", "")
    floor_price = nft_data.get("floor_price", 0)
    safelist_status = collection_meta.get("safelist_status", "not_requested")
    volume = collection_meta.get("stats", {}).get("total_volume", 0)
    is_spam_alchemy = (
        nft_data.get("contract", {}).get("isSpam", False)
        if isinstance(nft_data, dict) and "contract" in nft_data
        else False
    )
    return (
        is_spam_alchemy
        or not metadata
        or "http" not in image
        or "spam" in metadata
        or floor_price == 0
        or safelist_status == "not_requested"
        or volume < 10
    )


if __name__ == "__main__":
    load_dotenv()
    # Read preloaded collections and NFTs
    preloaded_file = "preloaded_nfts.txt"
    preloaded_nfts = {}
    if os.path.exists(preloaded_file):
        with open(preloaded_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        slug, nft_data = line.strip().split(":", 1)
                        nft_list = json.loads(nft_data)
                        if not isinstance(nft_list, list) or not all(
                            isinstance(nft, dict) for nft in nft_list
                        ):
                            print(f"Invalid preloaded data for {slug}: {nft_data}")
                            continue
                        preloaded_nfts[slug] = nft_list
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing preloaded data for {slug}: {e}")
                        continue

    # Use only the first NFT from each preloaded collection
    all_nfts_opensea = []
    for slug, nfts in preloaded_nfts.items():
        if nfts:
            all_nfts_opensea.append(nfts[0])  # Take only the first NFT

    if not all_nfts_opensea:
        print("No NFTs fetched from preloaded data. Check preloaded_nfts.txt.")
        sys.exit(1)
    else:
        print(f"Fetched {len(all_nfts_opensea)} NFTs from preloaded data.")
        print("Sample NFT:", all_nfts_opensea[0])

    # 7. Process and Combine Data
    valid_nfts = [nft for nft in all_nfts_opensea if isinstance(nft, dict)]
    if len(all_nfts_opensea) != len(valid_nfts):
        print(
            f"Warning: Filtered {len(all_nfts_opensea) - len(valid_nfts)} invalid NFTs (strings or non-dicts)"
        )
        for invalid in [nft for nft in all_nfts_opensea if not isinstance(nft, dict)]:
            print(f"Invalid NFT: {invalid}")
    if not valid_nfts:
        print("No valid NFTs to process. Exiting.")
        sys.exit(1)

    # Enhance with Alchemy data
    for i, nft in enumerate(valid_nfts):
        contract = nft.get("contract", {})
        token_id = nft.get("identifier", "")
        if contract and token_id:
            # Fetch floor price
            url = f"{ALCHEMY_BASE_URL}/getFloorPrice?contractAddress={contract}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    floor_data = response.json()
                    if "openSea" in floor_data:
                        valid_nfts[i]["floor_price"] = floor_data["openSea"][
                            "floorPrice"
                        ]  # Fixed key
                        valid_nfts[i]["price_currency"] = floor_data["openSea"][
                            "priceCurrency"
                        ]
                else:
                    print(f"Failed floor price for {contract}: {response.status_code}")
            except requests.RequestException as e:
                print(f"Network error for floor price {contract}: {e}")
            # Fetch metadata
            url = f"{ALCHEMY_BASE_URL}/getNFTMetadata?contractAddress={contract}&tokenId={token_id}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    meta_data = response.json()
                    if "openSeaMetadata" in meta_data["contract"]:
                        valid_nfts[i].update(
                            {
                                "collection_name": meta_data["contract"]["openSeaMetadata"][
                                    "collectionName"
                                ],
                                "description": meta_data["contract"]["openSeaMetadata"][
                                    "description"
                                ],
                                "total_supply": meta_data["contract"]["totalSupply"],
                            }
                        )
                else:
                    print(
                        f"Failed metadata for {contract}/{token_id}: {response.status_code}"
                    )
            except requests.RequestException as e:
                print(f"Network error for metadata {contract}/{token_id}: {e}")              
        data = {
            "owner_address": [
                nft.get("owner", {}) for nft in valid_nfts
            ],
            "contract_address": [
                nft.get("contract", {}) for nft in valid_nfts
            ],
            "token_id": [nft.get("identifier", "") for nft in valid_nfts],
            "token_type": [nft.get("token_standard", "ERC721") for nft in valid_nfts],
            "balance": [nft.get("balance", 1) for nft in valid_nfts],
            "collection_name": [nft.get("collection_name", "") for nft in valid_nfts],
            "collection_slug": [nft.get("collection", "") for nft in valid_nfts],
            "is_spam": [False] * len(valid_nfts),
            "floor_price": [nft.get("floor_price", 0) for nft in valid_nfts],
            "price_currency": [nft.get("price_currency", "ETH") for nft in valid_nfts],
            "description": [nft.get("description", "") for nft in valid_nfts],
            "last_ingested_at": [nft.get("updated_at", "") for nft in valid_nfts],
            "total_supply": [nft.get("total_supply", 0) for nft in valid_nfts],
            "image_url": [nft.get("image_url", "") for nft in valid_nfts],
        }

        df = pd.DataFrame(data)
        print(f"DataFrame {df} created with shape:", df.shape)

        # Apply spam filtering with collection metadata
        df["collection_meta"] = [
            fetch_collection_metadata(nft.get("collection", "")) for nft in valid_nfts
        ]
        df["is_spam"] = [
            is_spam(nft, meta) for nft, meta in zip(valid_nfts, df["collection_meta"])
        ]
        df = df[df["is_spam"] == False].reset_index(drop=True)

        # Clean up temporary column
        df = df.drop(columns=["collection_meta"])

        # Save raw and cleaned data
        df.to_csv("nft_data_raw.csv", index=False)
        df.to_csv("nft_data_cleaned.csv", index=False)
        print(f"Saved {len(df)} non-spam NFTs to nft_data_cleaned.csv")

    # 9. ML Model (Optional)
    if not df.empty and len(df) > 10:
        numeric_features = ["floor_price", "balance", "total_supply"]
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
