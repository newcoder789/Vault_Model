
url = "https://api.coingecko.com/api/v3/coins/bored-ape-yacht-club/history?date=30-12-2017&localization=true"

headers = {
    "accept": "application/json",
    "x-cg-api-key": "CG-4ZqWuHvqkpcRYLexcA5Ap1Ef",
}

response = requests.get(url, headers=headers)

print(response.text)