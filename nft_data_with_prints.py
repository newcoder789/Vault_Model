import requests
import json
import time
import os 
from dotenv import load_dotenv

load_dotenv()
# --- Configuration (Replace with your actual keys) ---
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")


def extract_contracts(filepath="preloaded_nfts.txt"):
    contracts = set()  # Using a set to avoid duplicates
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split into key and JSON string
                try:
                    key, json_array = line.split(":", 1)
                    nft_items = json.loads(json_array.strip())
                    for item in nft_items:
                        contract = item.get("contract")
                        if contract:
                            contracts.add(contract)
                except json.JSONDecodeError as jde:
                    print(f"JSON error: {jde}")
                except ValueError as ve:
                    print(f"Split error: {ve}")
    except Exception as e:
        print(f"Something went wrong: {e}")

    print(f"\n✅ Found {len(contracts)} unique contract(s):")
    for addr in contracts:
        print(addr)

    return list(contracts)


NFT_CONTRACT_ADDRESSES = extract_contracts()



# --- Data Storage ---
collected_data = []

# --- 1. Get Floor Prices and basic NFT data from Alchemy (more comprehensive) ---
print("Fetching data from Alchemy...")
for contract_address in NFT_CONTRACT_ADDRESSES:
    try:
        url = f"https://eth-mainnet.g.alchemy.com/nft/v2/{ALCHEMY_API_KEY}/getContractMetadata"
        params = {"contractAddress": contract_address}
        response = requests.get(url, params=params)
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        floor_price_url = (
            f"https://eth-mainnet.g.alchemy.com/nft/v2/{ALCHEMY_API_KEY}/getFloorPrice"
        )
        floor_price_params = {"contractAddress": contract_address}
        floor_price_response = requests.get(floor_price_url, params=floor_price_params)
        floor_price_response.raise_for_status()
        floor_price_data = floor_price_response.json()

        # Extract relevant info
        collection_info = {
            "contract_address": contract_address,
            "name": data.get("contractMetadata", {}).get("name"),
            "symbol": data.get("contractMetadata", {}).get("symbol"),
            "total_supply": data.get("contractMetadata", {}).get("totalSupply"),
            "floor_price_opensea": floor_price_data.get("openSea", {}).get(
                "floorPrice"
            ),
            "floor_price_looksrare": floor_price_data.get("looksrare", {}).get(
                "floorPrice"
            ),
            # You might want to get historical floor prices if available and time allows
        }
        collected_data.append(collection_info)
        print(f"  Fetched Alchemy data for {contract_address}")
        time.sleep(0.1)  # Be respectful to API rate limits
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching Alchemy data for {contract_address}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred for {contract_address} (Alchemy): {e}")


# --- 2. Get Collection Stats from OpenSea (for volume, owners etc.) ---
# Note: OpenSea API v2 uses collection_slug, not contract address directly for stats.
# You'll need to map contract addresses to slugs or manually get slugs for your chosen collections.
# For simplicity in this quick example, let's assume you have a mapping or will fetch slugs later.
# For now, let's use a dummy OpenSea call for demonstration or skip if you have enough from Alchemy.

print("\nFetching data from OpenSea (for collection stats)...")



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
                    print(f"⚠️ Error parsing line: {e}")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")

    print("\ncollection_slugs = {")
    for contract, slug in collection_slugs.items():
        print(f'    "{contract}": "{slug}",')
    print("}")

    return collection_slugs

collection_slugs = extract_collection_slugs()

for item in collected_data:  # Iterate over already collected data to add OpenSea stats
    contract_address = item["contract_address"]
    collection_slug = collection_slugs.get(contract_address)

    if collection_slug:
        try:
            url = f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats"
            headers = {"X-API-KEY": OPENSEA_API_KEY}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            stats_data = response.json()

            item["opensea_volume_all_time"] = stats_data.get("total", {}).get("volume")
            item["opensea_sales_all_time"] = stats_data.get("total", {}).get("sales")
            item["opensea_num_owners"] = stats_data.get("total", {}).get("num_owners")
            item["opensea_market_cap"] = stats_data.get("total", {}).get("market_cap")
            item["opensea_floor_price_24hr"] = stats_data.get("one_day", {}).get(
                "floor_price"
            )  # Note: This might be slightly different from the primary floor price as it's a 24hr snapshot
            item["opensea_volume_24hr"] = stats_data.get("one_day", {}).get("volume")
            item["opensea_sales_24hr"] = stats_data.get("one_day", {}).get("sales")
            print(f"  Fetched OpenSea stats for {collection_slug}")
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching OpenSea stats for {collection_slug}: {e}")
        except Exception as e:
            print(
                f"  An unexpected error occurred for {collection_slug} (OpenSea): {e}"
            )
    else:
        print(f"  Skipping OpenSea stats for {contract_address}: slug not found.")


# --- Save data to a JSON file ---
with open("nft_risk_data_raw.json", "w") as f:
    json.dump(collected_data, f, indent=4)
print("\nRaw data saved to nft_risk_data_raw.json")

# You'll need to manually inspect this data and potentially add a 'risk_label' column
# For example, 0 for low risk, 1 for medium, 2 for high, based on your knowledge
# or
# Define heuristics: e.g., if (24hr volume_change < -50% AND floor_price_change < -30%), then High Risk.
# This labeling is crucial for supervised machine learning.
