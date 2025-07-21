# import requests

# url = "https://api.coingecko.com/api/v3/search/trending?show_max=nfts"

# headers = {
#     "accept": "application/json",
#     "x-cg-api-key": "CG-4ZqWuHvqkpcRYLexcA5Ap1Ef",
# }

# response = requests.get(url, headers=headers)

# print(response.text)
# import requests

# url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true&status=active"

# headers = {
#     "accept": "application/json",
#     "x-cg-pro-api-key": "CG-4ZqWuHvqkpcRYLexcA5Ap1Ef",
# }

# response = requests.get(url, headers=headers)

# print(response.text)




 #  3 
 
# import requests

# url = "https://api.coingecko.com/api/v3/coins/pudgy-penguins/history?date=30-12-2017&localization=true"

# headers = {
#     "accept": "application/json",
#     "x-cg-api-key": "CG-4ZqWuHvqkpcRYLexcA5Ap1Ef",
# }

# response = requests.get(url, headers=headers)

# print(response.text)




#4 
import pandas as pd
import json

# --- 1. Load the provided CSV files ---
try:
    df_cleaned = pd.read_csv("nft_data_raw_combined.csv")
    df_for_ml = pd.read_csv("nft_data_cleaned_for_ml.csv")
    df_mid_drop = pd.read_csv("mid_drop_nft_data.csv")

    print("Successfully loaded all CSV files.")
except FileNotFoundError as e:
    print(
        f"Error: One of the CSV files was not found. Please ensure all required CSVs are in the working directory. {e}"
    )
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading CSV files: {e}")
    exit()

# --- 2. Standardize column names ---
# Rename columns for consistency across all DataFrames.
# We'll prioritize 'floor_price' from df_for_ml/df_mid_drop if it exists, as it might be the 'opensea_floor_price'.

# Rename columns in df_for_ml
df_for_ml = df_for_ml.rename(
    columns={
        "identifier": "token_id",
        "collection": "collection_slug",
        "contract": "contract_address",
        "opensea_floor_price": "floor_price_from_ml",  # Rename to avoid direct conflict and allow prioritization
    }
)

# Rename columns in df_mid_drop
df_mid_drop = df_mid_drop.rename(
    columns={
        "identifier": "token_id",
        "collection": "collection_slug",
        "contract": "contract_address",
        "opensea_floor_price": "floor_price_from_mid_drop",  # Rename to avoid direct conflict
    }
)

# Add a 'source_file' column to track where data came from (useful for debugging)
df_cleaned["source_file_cleaned"] = True
df_for_ml["source_file_for_ml"] = True
df_mid_drop["source_file_mid_drop"] = True

# --- 3. Merge DataFrames iteratively ---
# Start with df_cleaned as the base.
# Use 'contract_address' and 'token_id' as merge keys for NFT-level uniqueness.
# Use 'outer' merge to ensure no data is lost and all NFTs are included.

# Merge df_for_ml into df_cleaned
# Handle suffixes if column names overlap (e.g., 'description_x', 'description_y')
combined_df = pd.merge(
    df_cleaned,
    df_for_ml,
    on=["contract", "identifier"],
    how="outer",
    suffixes=("_cleaned", "_for_ml"),
)

# Merge df_mid_drop into the combined_df
combined_df = pd.merge(
    combined_df,
    df_mid_drop,
    on=["contract_address", "token_id"],
    how="outer",
    suffixes=("_combined", "_mid_drop"),
)

# --- Resolve conflicting columns after merges ---
# For columns that appeared in multiple original DataFrames (e.g., 'floor_price', 'description'),
# we need to consolidate them into a single column.
# Priority: df_for_ml/df_mid_drop (as they might have 'opensea_floor_price') -> df_cleaned.
# For descriptive text, take the first non-null. For numerical, take the non-null.

# Consolidate 'floor_price'
# Use 'floor_price_from_ml' or 'floor_price_from_mid_drop' if available and not NaN, else use 'floor_price_cleaned'
# Fill NaNs from right to left (more recent sources if available)
combined_df["floor_price"] = (
    combined_df["floor_price_from_ml"]
    .fillna(combined_df["floor_price_from_mid_drop"])
    .fillna(combined_df["floor_price_cleaned"])
)

# Drop the now redundant original floor price columns
combined_df.drop(
    columns=["floor_price_cleaned", "floor_price_from_ml", "floor_price_from_mid_drop"],
    errors="ignore",
    inplace=True,
)


# Consolidate 'description' (take first non-null)
if "description_for_ml" in combined_df.columns:
    combined_df["description"] = combined_df["description_cleaned"].fillna(
        combined_df["description_for_ml"]
    )
    combined_df.drop(
        columns=["description_cleaned", "description_for_ml"],
        errors="ignore",
        inplace=True,
    )
elif "description_cleaned" in combined_df.columns:
    combined_df.rename(columns={"description_cleaned": "description"}, inplace=True)


# Consolidate 'collection_name' and 'collection_slug'
# These are crucial for grouping and should be consistent. Take non-null from any source.
if "collection_name_for_ml" in combined_df.columns:
    combined_df["collection_name"] = combined_df["collection_name_cleaned"].fillna(
        combined_df["collection_name_for_ml"]
    )
    combined_df.drop(
        columns=["collection_name_cleaned", "collection_name_for_ml"],
        errors="ignore",
        inplace=True,
    )
elif "collection_name_cleaned" in combined_df.columns:
    combined_df.rename(
        columns={"collection_name_cleaned": "collection_name"}, inplace=True
    )

if "collection_slug_for_ml" in combined_df.columns:
    combined_df["collection_slug"] = combined_df["collection_slug_cleaned"].fillna(
        combined_df["collection_slug_for_ml"]
    )
    combined_df.drop(
        columns=["collection_slug_cleaned", "collection_slug_for_ml"],
        errors="ignore",
        inplace=True,
    )
elif "collection_slug_cleaned" in combined_df.columns:
    combined_df.rename(
        columns={"collection_slug_cleaned": "collection_slug"}, inplace=True
    )


# Consolidate 'total_supply'
if "total_supply_for_ml" in combined_df.columns:
    combined_df["total_supply"] = combined_df["total_supply_cleaned"].fillna(
        combined_df["total_supply_for_ml"]
    )
    combined_df.drop(
        columns=["total_supply_cleaned", "total_supply_for_ml"],
        errors="ignore",
        inplace=True,
    )
elif "total_supply_cleaned" in combined_df.columns:
    combined_df.rename(columns={"total_supply_cleaned": "total_supply"}, inplace=True)


# Clean up remaining suffixed columns (e.g., _cleaned, _for_ml, _combined, _mid_drop) that were not explicitly handled if they represent the same data
cols_to_drop_suffixes = [
    col
    for col in combined_df.columns
    if col.endswith(("_cleaned", "_for_ml", "_combined", "_mid_drop"))
    and col not in ["source_file_cleaned", "source_file_for_ml", "source_file_mid_drop"]
]
combined_df.drop(columns=cols_to_drop_suffixes, errors="ignore", inplace=True)


print(
    f"\nCombined DataFrame shape after merging and consolidating: {combined_df.shape}"
)
print("Combined DataFrame columns after consolidation:\n", combined_df.columns.tolist())
print("\nFirst 5 rows of Combined DataFrame after consolidation:")
print(combined_df.head())

# --- 4. Basic Data Cleaning and Type Conversion ---
numeric_features = [
    "floor_price",
    "total_supply",
    "opensea_market_cap",
    "opensea_volume_30D",
    "opensea_volume_all_time",
    "opensea_average_price",
    "opensea_sales_7D",
    "opensea_avg_price_7D",
    "opensea_volume_7D",
    "opensea_sales_30D",
    "opensea_sales_all_time",
    "opensea_sales_24hr",
    "opensea_avg_price_24hr",
    "opensea_avg_price_30D",
    "opensea_volume_24hr",
    "opensea_floor_price_24hr",
    "opensea_num_owners",
    "balance",
    "twitter_tweet_volume",
    "twitter_positive_tweets",
    "twitter_negative_tweets",
    "twitter_neutral_tweets",
]

for col in numeric_features:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(0)

# Drop 'Unnamed: 0' if it exists and is an artifact
if "Unnamed: 0" in combined_df.columns:
    combined_df = combined_df.drop(columns=["Unnamed: 0"])

datetime_cols = ["last_ingested_at", "updated_at"]
for col in datetime_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")

# --- 5. Feature Engineering (Basic Example) ---
if (
    "total_supply" in combined_df.columns
    and "opensea_num_owners" in combined_df.columns
    and "owner_ratio" not in combined_df.columns
):
    combined_df["owner_ratio"] = combined_df["opensea_num_owners"] / combined_df[
        "total_supply"
    ].replace(0, pd.NA)
    combined_df["owner_ratio"] = (
        combined_df["owner_ratio"].fillna(0).replace([float("inf"), -float("inf")], 0)
    )

if (
    "floor_price" in combined_df.columns
    and "opensea_market_cap" in combined_df.columns
    and "price_to_market_cap_ratio" not in combined_df.columns
):
    combined_df["price_to_market_cap_ratio"] = combined_df["floor_price"] / combined_df[
        "opensea_market_cap"
    ].replace(0, pd.NA)
    combined_df["price_to_market_cap_ratio"] = (
        combined_df["price_to_market_cap_ratio"]
        .fillna(0)
        .replace([float("inf"), -float("inf")], 0)
    )

print("\n--- Combined DataFrame Info after cleaning and basic feature engineering ---")
combined_df.info()

# --- 6. Store the processed data in a final CSV ---
output_csv_path = "final_nft_data_processed.csv"
combined_df.to_csv(output_csv_path, index=False)
print(f"\nProcessed and combined data saved to '{output_csv_path}'.")

# --- 7. Reiterate DappRadar Historical Data Limitation ---
print("\n--- Important Note on DappRadar Historical Data ---")
print(
    "The 'dappradar_historical_data.json' file you provided is the DappRadar API documentation, not the raw historical data."
)
print(
    "To get actual **yearly historical data** for NFTs (like detailed daily floor prices or volumes from DappRadar), you would need to:"
)
print("1. **Obtain an API key** from DappRadar.")
print(
    "2. **Use the DappRadar API endpoints** (as described in the documentation, e.g., `/nfts/collections` or `/dapps/{dappId}/history/{metric}`) to fetch the historical data programmatically."
)
print(
    "3. Once you retrieve that data, it can be loaded and integrated into this combined dataset for more comprehensive risk evaluation and charting of NFT price movements."
)
print(
    "This script has successfully combined and processed the CSV files you provided, but it does not include the actual historical data from DappRadar as it was not provided in a raw data format."
)