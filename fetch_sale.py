url = "https://api.opensea.io/v2/events"
sales = []

# Get creation date and total days

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
print( sales[:MAX_TOTAL_EVENTS])