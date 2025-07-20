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

# Load environment variables
load_dotenv()

# --- API Setup ---
open_sea_key = os.getenv("OPENSEA_API_KEY")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

BASE_URL_OPENSEA = "https://api.opensea.io/api/v2"
HEADERS_OPENSEA = {"X-API-KEY": open_sea_key, "accept": "application/json"}
ALCHEMY_BASE_URL = f"https://eth-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"

# --- Data Storage ---
collected_data = []


# --- Helper Functions from your existing code to extract contracts and slugs ---
def extract_contracts(filepath="preloaded_nfts.txt"):
    """Extracts unique contract addresses from the preloaded_nfts.txt file."""
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
    except Exception as e:
        print(f"An unexpected error occurred while extracting contracts: {e}")
    print(f"Found {len(contracts)} unique contract(s) from {filepath}.")
    return list(contracts)


def extract_collection_slugs(filepath="preloaded_nfts.txt"):
    """Extracts collection slugs mapped to contract addresses from preloaded_nfts.txt."""
    collection_slugs = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    collection_name, json_data = line.split(":", 1)
                    collection_name = (
                        collection_name.strip()
                    )  # This is likely the slug in your preloaded file
                    nft_items = json.loads(json_data.strip())
                    for item in nft_items:
                        contract = item.get("contract")
                        if contract and contract not in collection_slugs:
                            collection_slugs[contract] = (
                                collection_name  # Mapping contract to its slug
                            )
                except (ValueError, json.JSONDecodeError) as e:
                    print(
                        f"Error parsing line for collection slugs: {e} - Line: {line[:50]}..."
                    )
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while extracting collection slugs: {e}")
    print(f"Found {len(collection_slugs)} collection slugs from {filepath}.")
    return collection_slugs


# --- Fetching Functions from giga_fetch.py ---
def fetch_collections():  # Not directly used in the main loop, but kept for completeness
    url = "https://api.opensea.io/api/v2/collections?order_by=market_cap"
    headers = {"accept": "application/json", "x-api-key": open_sea_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch collections: {response.status_code} - {response.text}")
        return {"collections": []}
    return response.json()


def fetch_collection_nfts(
    slug, max_retries=3
):  # Not directly used in the main loop, but kept for completeness
    url = f"{BASE_URL_OPENSEA}/collection/{slug}/nfts"
    params = {"limit": 25}
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=HEADERS_OPENSEA, params=params, timeout=10
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


def fetch_alchemy_nfts(
    owner, max_retries=3
):  # Not directly used in the main loop, but kept for completeness
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


def fetch_collection_metadata(slug, max_retries=3):
    url = f"{BASE_URL_OPENSEA}/collections/{slug}"
    for attempt in range(max_retries):
        try:    
            response = requests.get(url, headers={"X-API-KEY": open_sea_key})
            print("response for fetch collection metadata", response.json())
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
    """Enhanced spam detection logic."""
    print(
        f"\n\n\n\nnftData:{nft_data}\n\n\n\n collection_meta:{collection_meta}\n\n\n\n"
    )
    # Safe conversion from string to dict
    if isinstance(collection_meta, str):
        try:
            collection_meta = ast.literal_eval(collection_meta)
        except Exception as e:
            print("Failed to parse collection_meta:", e)
            collection_meta = {}
    metadata = nft_data.get("description", "").lower()
    image = nft_data.get("image_url", "")
    floor_price = nft_data.get("floor_price", 0)
    safelist_status = collection_meta.get("safelist_status", "not_requested")
    volume = collection_meta.get("stats", {}).get("total_volume", 0)
    is_spam_alchemy = (
        nft_data.get("is_spam_alchemy_contract", False)
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


# --- Main Execution Block ---
if __name__ == "__main__":
    preloaded_file = "preloaded_nfts.txt"
    preloaded_nfts = {}
    
    # Read preloaded collections and NFTs
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
    else:
        print(
            f"Error: {preloaded_file} not found. Please create it with preloaded NFT data."
        )
        sys.exit(1)
    all_nfts_opensea = []
    for slug, nfts in preloaded_nfts.items():
        print(f"adding nft of {slug}: {nfts}")
        all_nfts_opensea.extend(nfts)  # Extend with all NFTs from the list

    if not all_nfts_opensea:
        print("No NFTs fetched from preloaded data. Check preloaded_nfts.txt.")
        sys.exit(1)
    else:
        print(f"Fetched {len(all_nfts_opensea)} NFTs from preloaded data.")

    # Filter out invalid NFT entries (non-dictionaries)
    valid_nfts = [nft for nft in all_nfts_opensea if isinstance(nft, dict)]
    print("final valid nfts:", valid_nfts)
    if len(all_nfts_opensea) != len(valid_nfts):
        print(
            f"Warning: Filtered {len(all_nfts_opensea) - len(valid_nfts)} invalid NFTs (non-dicts)."
        )
    if not valid_nfts:
        print("No valid NFTs to process. Exiting.")
        sys.exit(1)

    # Extract contract addresses and collection slugs for API calls
    NFT_CONTRACT_ADDRESSES = extract_contracts(preloaded_file)
    COLLECTION_SLUGS_MAP = extract_collection_slugs(preloaded_file)


    if not os.path.exists("mid_drop_nft_data.csv"):
        # Stores floor price and currency per collection
        collection_floor_data_cache = {} 
        # Stores general collection details (name, description, total supply, spam status)
        collection_details_cache = {} 
        # Stores OpenSea collection stats (volume, sales, etc.)
        opensea_collection_stats_cache = {}

        processed_nfts = []
        for i, nft in enumerate(valid_nfts):
            contract_address = nft.get("contract")
            token_id = nft.get("identifier")
            
            # Initialize NFT with existing data
            enhanced_nft = nft.copy()

            # Ensure contract_address and token_id are available for API calls
            if not (contract_address and token_id):
                print(f"Skipping NFT due to missing contract_address or token_id: {nft}")
                processed_nfts.append(enhanced_nft)
                continue # Skip to next NFT


            # --- Step 1: Fetch Floor Price and Currency (Optimized: once per collection) ---
            if contract_address not in collection_floor_data_cache:
                floor_price_url = f"{ALCHEMY_BASE_URL}/getFloorPrice?contractAddress={contract_address}"
                try:
                    response = requests.get(floor_price_url, timeout=10)
                    print(f"\n\n\nFor the nft-fetching floor price of {contract_address}: {response.json()}\n\n\n")
                    if response.status_code == 200:
                        floor_data = response.json()
                        current_floor_data = {}
                        if "openSea" in floor_data and "floorPrice" in floor_data["openSea"]:
                            current_floor_data["floor_price"] = floor_data["openSea"]["floorPrice"]
                            current_floor_data["price_currency"] = floor_data["openSea"]["priceCurrency"]
                        if "looksrare" in floor_data and "floorPrice" in floor_data["looksrare"]:
                            current_floor_data["looksrare_floor_price"] = floor_data["looksrare"]["floorPrice"]
                        collection_floor_data_cache[contract_address] = current_floor_data
                        print(f"\n\n\n FLOOR PRICE- {current_floor_data} \n\n\n")
                    else:
                        print(f"Failed Alchemy floor price for {contract_address}: {response.status_code} - {response.text}")
                        collection_floor_data_cache[contract_address] = {} # Cache empty to avoid retrying immediately
                except requests.RequestException as e:
                    print(f"Network error for Alchemy floor price {contract_address}: {e}")
                    collection_floor_data_cache[contract_address] = {} # Cache empty on error
                time.sleep(0.05) # Small delay to respect API rate limits

            # Apply cached floor price data to current NFT
            cached_floor_data = collection_floor_data_cache.get(contract_address, {})
            enhanced_nft["floor_price"] = cached_floor_data.get("floor_price", 0.0)
            enhanced_nft["price_currency"] = cached_floor_data.get("price_currency", "ETH")
            if "looksrare_floor_price" in cached_floor_data:
                enhanced_nft["looksrare_floor_price"] = cached_floor_data["looksrare_floor_price"]


            # --- Step 2: Fetch NFT Metadata (Individual and Collection-Level) ---
            # This call is *always* made per NFT to get its unique data (attributes, specific image, etc.)
            # but collection-level details from it will be cached.
            meta_url = f"{ALCHEMY_BASE_URL}/getNFTMetadata?contractAddress={contract_address}&tokenId={token_id}"
            try:
                response = requests.get(meta_url, timeout=10)
                if response.status_code == 200:
                    meta_data = response.json()

                    # Cache collection-level details if not already cached
                    if contract_address not in collection_details_cache:
                        collection_level_details = {}
                        contract_info = meta_data.get("contractMetadata", {})
                        opensea_meta = contract_info.get("openSea", {})
                        safe_info = meta_data.get("spamInfo",{})
                        
                        collection_level_details["collection_name"] = opensea_meta.get("collectionName", "")
                        # Storing collection description separately to differentiate from token's description
                        collection_level_details["description_collection"] = opensea_meta.get("description", "🥲")
                        collection_level_details["total_supply"] = opensea_meta.get("totalSupply", 0)
                        collection_level_details["collection_slug"] = opensea_meta.get("collectionSlug", "🥲")
                        collection_level_details["is_spam_alchemy_contract"] = safe_info.get("isSpam", False)
                        collection_details_cache[contract_address] = collection_level_details
                    # Apply cached collection-level details to current NFT
                    cached_collection_details = collection_details_cache.get(contract_address, {})
                    enhanced_nft["collection_name"] = cached_collection_details.get("collection_name", enhanced_nft.get("collection_name", ""))
                    # Use the collection's description as the primary 'description' field for consistency
                    enhanced_nft["description"] = cached_collection_details.get("description_collection", enhanced_nft.get("description", ""))
                    enhanced_nft["total_supply"] = cached_collection_details.get("total_supply", enhanced_nft.get("total_supply", 0))
                    enhanced_nft["is_spam_alchemy_contract"] = cached_collection_details.get("is_spam_alchemy_contract", False)
                    enhanced_nft["collection_slug"] = cached_collection_details.get("collection_slug", "🥲")

                    #individual data 
                    nft_info = meta_data.get("id", {})
                    media = meta_data.get("media", {})
                    enhanced_nft["name"] = nft_info.get("tokenId", enhanced_nft.get("name", ""))
                    if meta_data.get("description"): 
                        enhanced_nft["description"] = meta_data.get("description")
                    if meta_data.get("metadata"):
                        enhanced_nft["metadata"] = meta_data.get("metadata", {})
                    if media:
                        enhanced_nft["image_url"] = media["thumbnail"]
                else:
                    print(f"Failed Alchemy metadata for {contract_address}/{token_id}: {response.status_code} - {response.text}")
            except requests.RequestException as e:
                print(f"Network error for Alchemy metadata {contract_address}/{token_id}: {e}")
            time.sleep(0.05) # Small delay for API calls


            # --- Step 3: Fetch OpenSea Collection Stats (Optimized: once per collection) ---
            collection_slug_for_stats = COLLECTION_SLUGS_MAP.get(contract_address)
            if collection_slug_for_stats:
                if collection_slug_for_stats not in opensea_collection_stats_cache:
                    try:
                        url = f"https://api.opensea.io/api/v2/collections/{collection_slug_for_stats}/stats"
                        headers = {"X-API-KEY": open_sea_key}
                        response = requests.get(url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            stats_data = response.json()
                            intervals = stats_data.get("intervals",[])
                            
                            interval1 = intervals[0]
                            interval2 = intervals[1]
                            interval3 = intervals[2]
                            
                            opensea_stats = {
                                "opensea_volume_all_time": stats_data.get("total", {}).get("volume"),
                                "opensea_sales_all_time": stats_data.get("total", {}).get("sales"),
                                "opensea_num_owners": stats_data.get("total", {}).get("num_owners"),
                                "opensea_market_cap": stats_data.get("total", {}).get("market_cap"),
                                "opensea_floor_price": stats_data.get("total", {}).get("floor_price"),
                                "opensea_average_price": stats_data.get("total", {}).get("average_price"),
                                "opensea_volume_24hr": interval1.get("volume"),
                                "opensea_sales_24hr": interval1.get("sales"),
                                "opensea_avg_price_24hr": interval1.get("average_price"),
                                "opensea_volume_7D": interval2.get("volume"),
                                "opensea_sales_7D": interval2.get("sales"),
                                "opensea_avg_price_7D": interval2.get("average_price"), 
                                "opensea_volume_30D": interval3.get("volume"),
                                "opensea_sales_30D": interval3.get("sales"),
                                "opensea_avg_price_30D": interval3.get("average_price"),
                            }
                            opensea_collection_stats_cache[collection_slug_for_stats] = opensea_stats
                        else:
                            print(f"Failed OpenSea stats for {collection_slug_for_stats}: {response.status_code} - {response.text}")
                            opensea_collection_stats_cache[collection_slug_for_stats] = {} # Cache empty on error
                    except requests.RequestException as e:
                        print(f"Network error for OpenSea stats {collection_slug_for_stats}: {e}")
                        opensea_collection_stats_cache[collection_slug_for_stats] = {} # Cache empty on error
                    time.sleep(0.05) # Small delay for API calls
                # Apply cached OpenSea stats to current NFT
                cached_opensea_stats = opensea_collection_stats_cache.get(collection_slug_for_stats, {})
                enhanced_nft["opensea_volume_all_time"] = cached_opensea_stats.get("opensea_volume_all_time")
                enhanced_nft["opensea_sales_all_time"] = cached_opensea_stats.get("opensea_sales_all_time")
                enhanced_nft["opensea_num_owners"] = cached_opensea_stats.get("opensea_num_owners")
                enhanced_nft["opensea_market_cap"] = cached_opensea_stats.get("opensea_market_cap")
                enhanced_nft["opensea_floor_price_24hr"] = cached_opensea_stats.get("opensea_floor_price_24hr")
                enhanced_nft["opensea_volume_24hr"] = cached_opensea_stats.get("opensea_volume_24hr")
                enhanced_nft["opensea_sales_24hr"] = cached_opensea_stats.get("opensea_sales_24hr")
                enhanced_nft["opensea_avg_price_24hr"] = cached_opensea_stats.get("opensea_avg_price_24hr")                
                enhanced_nft["opensea_volume_7D"] = cached_opensea_stats.get("opensea_volume_7D")
                enhanced_nft["opensea_sales_7D"] = cached_opensea_stats.get("opensea_sales_7D")
                enhanced_nft["opensea_avg_price_7D"] = cached_opensea_stats.get("opensea_avg_price_7D")
                enhanced_nft["opensea_volume_30D"] = cached_opensea_stats.get("opensea_volume_30D")
                enhanced_nft["opensea_sales_30D"] = cached_opensea_stats.get("opensea_sales_30D")
                enhanced_nft["opensea_avg_price_30D"] = cached_opensea_stats.get("opensea_avg_price_30D")
                enhanced_nft["opensea_floor_price"] = cached_opensea_stats.get("opensea_floor_price")
                enhanced_nft["opensea_average_price"] = cached_opensea_stats.get("opensea_average_price")
            else:
                print(f"Skipping OpenSea stats for {contract_address}: slug not found in map.")                             
            processed_nfts.append(enhanced_nft)

        # Example of what processed_nfts might look like after enhancement
        print("\n--- Processed NFTs (Example Output) ---")
        for nft in processed_nfts:
            print(f"Identifier: {nft.get('identifier')}, "
                f"Collection: {nft.get('collection_name')}, "
                f"Floor Price: {nft.get('floor_price')} {nft.get('price_currency')}, "
                f"Total Supply: {nft.get('total_supply')}, "
                f"Is Spam (Alchemy): {nft.get('is_spam_alchemy_contract')}")
            time.sleep(0.05)  # Small delay for other API calls

        # Create DataFrame
        # Ensure all possible keys are present in the dictionary for DataFrame creation
        all_keys = set()
        for nft in processed_nfts:
            all_keys.update(nft.keys())

        data_for_df = {}
        for key in all_keys:
            data_for_df[key] = [nft.get(key) for nft in processed_nfts]

        df = pd.DataFrame(data_for_df)
        df.to_csv("early_drop_nft_data.csv")
        print(f"DataFrame created with shape: {df.shape}")
        print("creating collection field....", df.iterrows())
        df["collection_meta"] = [
            fetch_collection_metadata(COLLECTION_SLUGS_MAP.get(row["contract"]))
            for index, row in df.iterrows()
        ]
        df.to_csv("mid_drop_nft_data.csv")
    else: 
        print("reading file....")
        df = pd.read_csv("mid_drop_nft_data.csv")
    
    print("collection_metadata field  createed.....")
    df = df[df["collection_meta"].apply(lambda x: bool(x))].reset_index(drop=True)
    if df.empty:
        print(
            "No valid NFTs remaining after filtering by collection metadata. Exiting."
        )
        sys.exit(1)
    print("now checking is spam.....")
    # Use the 'is_spam' function on the filtered DataFrame
    df["is_spam"] = [
        is_spam(nft_row, meta_item)
        for nft_row, meta_item in zip(
            df.to_dict(orient="records"), df["collection_meta"]
        )
    ]
    df = df.drop(columns=["collection_meta"])  # Clean up temporary column

    df.to_csv("nft_data_raw_combined.csv", index=False)
    print("Raw combined NFT data saved to nft_data_raw_combined.csv")

    # Filter out spam NFTs before saving cleaned data for ML
    df_cleaned = df[df["is_spam_alchemy_contract"] == False].reset_index(drop=True)
    df_cleaned.to_csv("nft_data_cleaned_for_ml.csv", index=False)
    print(f"Saved {len(df_cleaned)} non-spam NFTs to nft_data_cleaned_for_ml.csv")

    # --- Machine Learning Model ---
    if not df_cleaned.empty and len(df_cleaned) > 10:
        # Prepare features for the ML model
        # Select relevant numeric features (ensure they exist and are numeric)
        numeric_features = [
            "floor_price",
            "looksrare_floor_price",
            "balance",
            "total_supply",
            "opensea_volume_all_time",
            "opensea_sales_all_time",
            "opensea_num_owners",
            "opensea_market_cap",
            "opensea_floor_price_24hr",
            "opensea_volume_24hr",
            "opensea_sales_24hr",
            "opensea_avg_price_24hr",
        ]
        # Filter features that are actually in the DataFrame
        numeric_features = [f for f in numeric_features if f in df_cleaned.columns]
        for col in numeric_features:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)

        X = df_cleaned[numeric_features].copy()

        # Encode categorical variables
        categorical_cols = [
            "token_standard",
            "price_currency",  # token_standard from NFT, price_currency from Alchemy
        ]
        # Ensure categorical columns exist and fill NaNs for encoder
        for col in categorical_cols:
            if col not in df_cleaned.columns:
                df_cleaned[col] = ""  # Add empty string if column missing
            df_cleaned[col] = df_cleaned[col].fillna("unknown")  # Fill any NaNs

        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )  # handle_unknown to prevent errors on unseen categories
        encoded_cols = encoder.fit_transform(df_cleaned[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_cols, columns=encoder.get_feature_names_out(categorical_cols)
        )

        # Concatenate numerical and encoded categorical features
        X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

        # The target 'is_spam' is already binary, which is a good start for a basic classifier.
        # For a multi-class risk model (low/medium/high), you'd replace 'is_spam' with your
        # manually created 'risk_label' column after you've done that data labeling.
        y = df_cleaned["is_spam"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        print(f"\nModel Accuracy (Spam Detection): {accuracy}")

        # Save the trained model and encoder
        import joblib

        joblib.dump(model, "nft_spam_classifier_model.pkl")
        joblib.dump(encoder, "nft_categorical_encoder.pkl")
        print("Trained spam classifier model and encoder saved.")

    else:
        print("Not enough non-spam data to train model.")
