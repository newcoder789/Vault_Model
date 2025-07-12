# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import json
# # COLLECTIONS = ["bored-ape-yacht-club", "cryptopunks", "azuki"]
# # COLLECTIONS = "0xf54c9a0e44a5f5afd27c7ac8a176a843b9114f1d"
# # url = f"https://api.opensea.io/api/v2/collection/{COLLECTIONS}"
# # url = "https://api.opensea.io/api/v2/collections"
# # url = "https://api.opensea.io/api/v2/chain/{chain}/account/{address}/nfts"
"""
Get Collections
"""
# import requests
# url = "https://api.opensea.io/api/v2/collections"
# headers = {"accept": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.text)


"""
Get NFTs (by collection)
"""
# import requests
# url = "https://api.opensea.io/api/v2/collection/boredapeyachtclub/nfts"
# headers = {"accept": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.text)


"""
Get NFT
get
https://api.opensea.io/api/v2/chain/{chain}/contract/{address}/nfts/{identifier}
Get metadata, traits, ownership information, and rarity for a single NFT.
"""
# import requests
# url = "https://api.opensea.io/api/v2/chain/ethereum/contract/0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d/nfts/9989"
# headers = {"accept": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.text)


"""
Get Events (by NFT)
get
https://api.opensea.io/api/v2/events/chain/{chain}/contract/{address}/nfts/{identifier}
Get a list of events for a single NFT.
"""
# import requests
# url = "https://api.opensea.io/api/v2/events/chain/ethereum/contract/0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d/nfts/9989"
# headers = {"accept": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.text)



# url = "https://api.opensea.io/api/v2/collections?order_by=market_cap"
# headers = {"accept": "application/json"}

# response = requests.get(url, headers=headers)
# data = response.json()["collections"]

# session = requests.Session()
# retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
# session.mount("https://", HTTPAdapter(max_retries=retries))
# all_nfts = []
# for collection in data:
#     collection_name = collection["collection"]
#     url = f"https://api.opensea.io/api/v2/collection/{collection_name}/nfts"

#     headers = {
#         "accept": "application/json",
#         "User-Agent": "MyNFTDataCollector/1.0",
#     }
#     try:
#         print(f"Fetching NFTs for collection: {collection_name}...")
#         response = session.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         nft_data = response.json()
#         # Extend the main list with the NFTs from the current collection
#         all_nfts.extend(nft_data.get("nfts", []))
#         print(f"  > Found and added {len(nft_data.get('nfts', []))} NFTs.")
        
#     except requests.exceptions.HTTPError as http_err:
#         status_code = http_err.response.status_code
#         if status_code == 404:
#             print(f"  > Collection '{collection_name}' not found (404). Skipping. 🥸")

# # After the loop, write the entire list of NFTs to the file once.
# print(f"\nFinished fetching. Total NFTs collected: {len(all_nfts)}.")
# print("Writing all collected NFTs to nfts.json...")
# with open("./nfts.json", "w", encoding="utf-8") as f:
#     json.dump(all_nfts, f, indent=4)
# print("Successfully wrote to nfts.json.")







"""

Fetch sale events from OpenSea


"""
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
# Set up requests with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY")

DAYS = 30
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
    try:
        while True:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            sales.extend(data.get("asset_events", []))
            if "next" not in data or not data["next"]:
                break
            params["cursor"] = data["next"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching events for {collection_slug}: {e}")
    return sales