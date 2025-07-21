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
import ast
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
open_sea_key = os.getenv("OPENSEA_API_KEY")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
BASE_URL_OPENSEA = "https://api.opensea.io/api/v2"
HEADERS_OPENSEA = {"X-API-KEY": open_sea_key, "accept": "application/json"}
ALCHEMY_BASE_URL = f"https://eth-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"


# Helper functions
def extract_contracts(filepath="preloaded_nfts.txt"):
    contracts = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    _, json_array = line.split(":", 1)
                    nft_items = json.loads(json_array.strip())
                    for item in nft_items:
                        contract = item.get("contract")
                        if contract:
                            contracts.add(contract)
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"Error parsing line in preloaded_nfts.txt: {e} - Line: {line[:50]}..."
                    )
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it exists.")
        sys.exit(1)
    print(f"Found {len(contracts)} unique contract(s) from {filepath}.")
    return list(contracts)


def extract_collection_slugs(filepath="preloaded_nfts.txt"):
    collection_slugs = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    collection_name, json_data = line.split(":", 1)
                    collection_name = collection_name.strip()
                    nft_items = json.loads(json_data.strip())
                    for item in nft_items:
                        contract = item.get("contract")
                        if contract and contract not in collection_slugs:
                            collection_slugs[contract] = collection_name
                except (ValueError, json.JSONDecodeError) as e:
                    print(
                        f"Error parsing line for collection slugs: {e} - Line: {line[:50]}..."
                    )
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it exists.")
        sys.exit(1)
    print(f"Found {len(collection_slugs)} collection slugs from {filepath}.")
    return collection_slugs


def fetch_collection_metadata(slug, max_retries=3):
    if not slug:
        return {}
    url = f"{BASE_URL_OPENSEA}/collections/{slug}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS_OPENSEA, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"Attempt {attempt + 1} failed for {slug} metadata: {response.status_code} - {response.text}"
                )
                break
        except requests.RequestException as e:
            print(
                f"Network error for {slug} metadata: {e}. Attempt {attempt + 1} of {max_retries}"
            )
            time.sleep(2**attempt)
    return {}


def is_spam(nft_data, collection_meta):
    """Relaxed spam detection logic."""
    if isinstance(collection_meta, str):
        try:
            collection_meta = ast.literal_eval(collection_meta)
        except Exception as e:
            print(f"Failed to parse collection_meta: {e}")
            collection_meta = {}

    is_spam_alchemy = nft_data.get("is_spam_alchemy_contract", False)
    metadata = nft_data.get("description", "").lower()

    spam_conditions = [is_spam_alchemy, "spam" in metadata and metadata != ""]
    is_spam_result = any(spam_conditions)
    # print(
    #     f"NFT {nft_data.get('identifier', 'unknown')}: is_spam={is_spam_result}, "
    #     f"alchemy_spam={is_spam_alchemy}, metadata={bool(metadata)}"
    # )
    return is_spam_result


def calculate_price_movement(nft):
    """Calculate price movement trend and assign a label."""
    price_24hr = nft.get("opensea_avg_price_24hr", 0.0)
    price_7d = nft.get("opensea_avg_price_7D", 0.0)

    if price_7d == 0 or price_24hr == 0:
        return "stable", 0.0

    percentage_change = ((price_24hr - price_7d) / price_7d) * 100

    if percentage_change > 20:
        return "big_up", percentage_change
    elif 5 <= percentage_change <= 20:
        return "small_up", percentage_change
    elif -5 <= percentage_change < 5:
        return "stable", percentage_change
    elif -20 <= percentage_change < -5:
        return "small_down", percentage_change
    else:
        return "big_down", percentage_change


# Main Execution Block
if __name__ == "__main__":
    preloaded_file = "preloaded_nfts.txt"
    preloaded_nfts = {}

    # Validate preloaded_nfts.txt
    if not os.path.exists(preloaded_file):
        print(
            f"Error: {preloaded_file} not found. Please create it with valid NFT data."
        )
        sys.exit(1)

    with open(preloaded_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    slug, nft_data = line.strip().split(":", 1)
                    nft_list = json.loads(nft_data)
                    if not isinstance(nft_list, list) or not all(
                        isinstance(nft, dict) for nft in nft_list
                    ):
                        print(f"Invalid preloaded data for {slug}: {nft_data[:50]}...")
                        continue
                    preloaded_nfts[slug] = nft_list
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing preloaded data for {slug}: {e}")
                    continue

    all_nfts_opensea = []
    for slug, nfts in preloaded_nfts.items():
        print(f"Adding NFTs for {slug}: {len(nfts)} items")
        all_nfts_opensea.extend(nfts)

    if not all_nfts_opensea:
        print("No NFTs fetched from preloaded data. Check preloaded_nfts.txt.")
        sys.exit(1)

    valid_nfts = [nft for nft in all_nfts_opensea if isinstance(nft, dict)]
    print(f"Total NFTs: {len(all_nfts_opensea)}, Valid NFTs: {len(valid_nfts)}")
    if len(all_nfts_opensea) != len(valid_nfts):
        print(
            f"Warning: Filtered {len(all_nfts_opensea) - len(valid_nfts)} invalid NFTs."
        )
    if not valid_nfts:
        print("No valid NFTs to process. Exiting.")
        sys.exit(1)

    NFT_CONTRACT_ADDRESSES = extract_contracts(preloaded_file)
    COLLECTION_SLUGS_MAP = extract_collection_slugs(preloaded_file)

    if not os.path.exists("mid_drop_nft_data.csv"):
        collection_floor_data_cache = {}
        collection_details_cache = {}
        opensea_collection_stats_cache = {}
        processed_nfts = []

        for i, nft in enumerate(valid_nfts):
            contract_address = nft.get("contract")
            token_id = nft.get("identifier")
            enhanced_nft = nft.copy()

            if not (contract_address and token_id):
                print(
                    f"Skipping NFT {i + 1}: missing contract_address or token_id: {nft}"
                )
                processed_nfts.append(enhanced_nft)
                continue

            # Fetch Floor Price
            if contract_address not in collection_floor_data_cache:
                floor_price_url = f"{ALCHEMY_BASE_URL}/getFloorPrice?contractAddress={contract_address}"
                try:
                    response = requests.get(floor_price_url, timeout=10)
                    if response.status_code == 200:
                        floor_data = response.json()
                        current_floor_data = {
                            "floor_price": floor_data.get("openSea", {}).get(
                                "floorPrice", 0.0
                            ),
                            "price_currency": floor_data.get("openSea", {}).get(
                                "priceCurrency", "ETH"
                            ),
                            "looksrare_floor_price": floor_data.get(
                                "looksrare", {}
                            ).get("floorPrice", 0.0),
                        }
                        collection_floor_data_cache[contract_address] = (
                            current_floor_data
                        )
                    else:
                        print(
                            f"Failed Alchemy floor price for {contract_address}: {response.status_code}"
                        )
                        collection_floor_data_cache[contract_address] = {}
                except requests.RequestException as e:
                    print(
                        f"Network error for Alchemy floor price {contract_address}: {e}"
                    )
                    collection_floor_data_cache[contract_address] = {}
                time.sleep(0.1)

            cached_floor_data = collection_floor_data_cache.get(contract_address, {})
            enhanced_nft.update(cached_floor_data)

            # Fetch NFT Metadata
            meta_url = f"{ALCHEMY_BASE_URL}/getNFTMetadata?contractAddress={contract_address}&tokenId={token_id}"
            try:
                response = requests.get(meta_url, timeout=10)
                if response.status_code == 200:
                    meta_data = response.json()
                    if contract_address not in collection_details_cache:
                        collection_level_details = {
                            "collection_name": meta_data.get("contractMetadata", {})
                            .get("openSea", {})
                            .get("collectionName", ""),
                            "description_collection": meta_data.get(
                                "contractMetadata", {}
                            )
                            .get("openSea", {})
                            .get("description", ""),
                            "total_supply": meta_data.get("contractMetadata", {})
                            .get("openSea", {})
                            .get("totalSupply", 0),
                            "collection_slug": meta_data.get("contractMetadata", {})
                            .get("openSea", {})
                            .get("collectionSlug", ""),
                            "is_spam_alchemy_contract": meta_data.get(
                                "spamInfo", {}
                            ).get("isSpam", False),
                        }
                        collection_details_cache[contract_address] = (
                            collection_level_details
                        )
                    enhanced_nft.update(
                        collection_details_cache.get(contract_address, {})
                    )
                    enhanced_nft["name"] = meta_data.get("id", {}).get(
                        "tokenId", enhanced_nft.get("name", "")
                    )
                    if meta_data.get("description"):
                        enhanced_nft["description"] = meta_data.get("description")
                    if meta_data.get("metadata"):
                        enhanced_nft["metadata"] = meta_data.get("metadata", {})
                    if meta_data.get("media", {}):
                        enhanced_nft["image_url"] = meta_data.get("media", {}).get(
                            "thumbnail", ""
                        )
                else:
                    print(
                        f"Failed Alchemy metadata for {contract_address}/{token_id}: {response.status_code}"
                    )
            except requests.RequestException as e:
                print(
                    f"Network error for Alchemy metadata {contract_address}/{token_id}: {e}"
                )
            time.sleep(0.1)

            # Fetch OpenSea Collection Stats
            collection_slug_for_stats = COLLECTION_SLUGS_MAP.get(contract_address)
            if collection_slug_for_stats:
                if collection_slug_for_stats not in opensea_collection_stats_cache:
                    try:
                        url = f"{BASE_URL_OPENSEA}/collections/{collection_slug_for_stats}/stats"
                        response = requests.get(
                            url, headers=HEADERS_OPENSEA, timeout=10
                        )
                        if response.status_code == 200:
                            stats_data = response.json()
                            intervals = stats_data.get("intervals", [])
                            interval1 = intervals[0] if intervals else {}
                            interval2 = intervals[1] if len(intervals) > 1 else {}
                            interval3 = intervals[2] if len(intervals) > 2 else {}
                            opensea_stats = {
                                "opensea_volume_all_time": stats_data.get(
                                    "total", {}
                                ).get("volume", 0),
                                "opensea_sales_all_time": stats_data.get(
                                    "total", {}
                                ).get("sales", 0),
                                "opensea_num_owners": stats_data.get("total", {}).get(
                                    "num_owners", 0
                                ),
                                "opensea_market_cap": stats_data.get("total", {}).get(
                                    "market_cap", 0
                                ),
                                "opensea_floor_price": stats_data.get("total", {}).get(
                                    "floor_price", 0
                                ),
                                "opensea_average_price": stats_data.get(
                                    "total", {}
                                ).get("average_price", 0),
                                "opensea_volume_24hr": interval1.get("volume", 0),
                                "opensea_sales_24hr": interval1.get("sales", 0),
                                "opensea_avg_price_24hr": interval1.get(
                                    "average_price", 0
                                ),
                                "opensea_volume_7D": interval2.get("volume", 0),
                                "opensea_sales_7D": interval2.get("sales", 0),
                                "opensea_avg_price_7D": interval2.get(
                                    "average_price", 0
                                ),
                                "opensea_volume_30D": interval3.get("volume", 0),
                                "opensea_sales_30D": interval3.get("sales", 0),
                                "opensea_avg_price_30D": interval3.get(
                                    "average_price", 0
                                ),
                            }
                            opensea_collection_stats_cache[
                                collection_slug_for_stats
                            ] = opensea_stats
                        else:
                            print(
                                f"Failed OpenSea stats for {collection_slug_for_stats}: {response.status_code}"
                            )
                            opensea_collection_stats_cache[
                                collection_slug_for_stats
                            ] = {}
                    except requests.RequestException as e:
                        print(
                            f"Network error for OpenSea stats {collection_slug_for_stats}: {e}"
                        )
                        opensea_collection_stats_cache[collection_slug_for_stats] = {}
                    time.sleep(0.1)
                enhanced_nft.update(
                    opensea_collection_stats_cache.get(collection_slug_for_stats, {})
                )

            processed_nfts.append(enhanced_nft)

        # Create DataFrame
        all_keys = set()
        for nft in processed_nfts:
            all_keys.update(nft.keys())
        data_for_df = {
            key: [nft.get(key) for nft in processed_nfts] for key in all_keys
        }
        df = pd.DataFrame(data_for_df)
        df.to_csv("early_drop_nft_data.csv", index=False)
        print(f"DataFrame created with shape: {df.shape}")

        # Add collection metadata
        df["collection_meta"] = [
            fetch_collection_metadata(COLLECTION_SLUGS_MAP.get(row["contract"], ""))
            for _, row in df.iterrows()
        ]
        df.to_csv("mid_drop_nft_data.csv", index=False)
    else:
        print("Reading existing mid_drop_nft_data.csv...")
        df = pd.read_csv("mid_drop_nft_data.csv")

    # Debug collection metadata filtering
    print(f"Before metadata filter: {df.shape[0]} NFTs")
    df["has_valid_meta"] = df["collection_meta"].apply(lambda x: bool(x))
    print(f"NFTs with valid collection_meta: {df['has_valid_meta'].sum()}")
    df = df[df["has_valid_meta"]].reset_index(drop=True)
    print(f"After metadata filter: {df.shape[0]} NFTs")
    if df.empty:
        print(
            "No NFTs remaining after collection metadata filter. Check API responses or slugs in preloaded_nfts.txt."
        )
        sys.exit(1)

    # Apply spam detection
    df["is_spam"] = [
        is_spam(nft_row, meta_item)
        for nft_row, meta_item in zip(
            df.to_dict(orient="records"), df["collection_meta"]
        )
    ]
    df = df.drop(columns=["collection_meta", "has_valid_meta"])

    # Calculate price movement labels
    df[["price_movement_label", "percentage_change"]] = [
        calculate_price_movement(nft_row) for nft_row in df.to_dict(orient="records")
    ]

    df.to_csv("nft_data_raw_combined.csv", index=False)
    print("Raw combined NFT data saved to nft_data_raw_combined.csv")

    # Filter non-spam NFTs
    df_cleaned = df[df["is_spam"] == False].reset_index(drop=True)
    print(f"Non-spam NFTs: {len(df_cleaned)}")
    df_cleaned.to_csv("nft_data_cleaned_for_ml.csv", index=False)

    # Visualization: Overall and Per-Collection Price Movement Distribution
    if not df_cleaned.empty:
        # Overall distribution
        price_movement_counts = df_cleaned["price_movement_label"].value_counts()
        print("\nOverall Price Movement Distribution:")
        print(price_movement_counts)

        # Create overall bar chart
        labels = ["big_up", "small_up", "stable", "small_down", "big_down"]
        counts = [price_movement_counts.get(label, 0) for label in labels]
        colors = ["#2ca02c", "#98df8a", "#d3d3d3", "#ff9896", "#d62728"]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color=colors, edgecolor="black")
        plt.title("Overall NFT Price Movement Distribution")
        plt.xlabel("Price Movement Category")
        plt.ylabel("Number of NFTs")
        plt.tight_layout()
        plt.savefig("nft_price_movement_overall.png")
        plt.close()
        print(
            "Overall price movement distribution chart saved as nft_price_movement_overall.png"
        )

        # Per-collection distribution
        if "collection_name" in df_cleaned.columns:
            # Group by collection_name and price_movement_label
            collection_counts = (
                df_cleaned.groupby(["collection_name", "price_movement_label"])
                .size()
                .unstack(fill_value=0)
            )
            collections = collection_counts.index
            n_collections = len(collections)

            if n_collections > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                bar_width = 0.15
                index = range(len(collections))

                for i, label in enumerate(labels):
                    counts = [
                        collection_counts.get(
                            label, pd.Series(0, index=collections)
                        ).loc[coll]
                        for coll in collections
                    ]
                    plt.bar(
                        [x + bar_width * i for x in index],
                        counts,
                        bar_width,
                        label=label,
                        color=colors[i],
                        edgecolor="black",
                    )

                plt.xlabel("Collection")
                plt.ylabel("Number of NFTs")
                plt.title("Price Movement Distribution by Collection")
                plt.xticks([x + bar_width * 2 for x in index], collections, rotation=45)
                plt.legend(labels)
                plt.tight_layout()
                plt.savefig("nft_price_movement_by_collection.png")
                plt.close()
                print(
                    "Per-collection price movement distribution chart saved as nft_price_movement_by_collection.png"
                )
            else:
                print("No collections available for per-collection visualization.")
        else:
            print("No collection_name column for per-collection visualization.")
    else:
        print("No non-spam NFTs to visualize. Check spam filter or input data.")

    # Train ML model
    if not df_cleaned.empty and len(df_cleaned) > 10:
        numeric_features = [
            "floor_price",
            "looksrare_floor_price",
            "total_supply",
            "opensea_volume_all_time",
            "opensea_sales_all_time",
            "opensea_num_owners",
            "opensea_market_cap",
            "opensea_volume_24hr",
            "opensea_sales_24hr",
            "opensea_avg_price_24hr",
            "opensea_volume_7D",
            "opensea_sales_7D",
            "opensea_avg_price_7D",
            "opensea_volume_30D",
            "opensea_sales_30D",
            "opensea_avg_price_30D",
            "percentage_change",
        ]
        numeric_features = [f for f in numeric_features if f in df_cleaned.columns]
        for col in numeric_features:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)

        X = df_cleaned[numeric_features].copy()

        categorical_cols = ["token_standard", "price_currency"]
        for col in categorical_cols:
            if col not in df_cleaned.columns:
                df_cleaned[col] = ""
            df_cleaned[col] = df_cleaned[col].fillna("unknown")

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_cols = encoder.fit_transform(df_cleaned[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_cols, columns=encoder.get_feature_names_out(categorical_cols)
        )

        X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)
        y = df_cleaned["price_movement_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"\nModel Accuracy (Price Movement Prediction): {accuracy}")
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, "nft_price_movement_classifier_model.pkl")
        joblib.dump(encoder, "nft_price_movement_categorical_encoder.pkl")
        print("Trained price movement classifier model and encoder saved.")
    else:
        print(
            f"Not enough non-spam data to train model. Need at least 10 NFTs, got {len(df_cleaned)}."
        )
