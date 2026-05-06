import pandas as pd
import requests

url = "https://data.ny.gov/resource/5wq4-mkjj.csv"

query = """
SELECT
    transit_timestamp,
    transit_mode,
    station_complex_id,
    station_complex,
    borough,
    payment_method,
    fare_class_category,
    ridership,
    transfers,
    latitude,
    longitude,
    georeference
WHERE borough = 'Manhattan'
AND transit_timestamp >= '2026-01-01T00:00:00'
AND transit_timestamp <= '2026-05-06T23:59:59'
AND date_extract_hh(transit_timestamp) IN (7,8,9,10)
LIMIT 500000
"""

response = requests.get(url, params={"$query": query})

print(response.status_code)
print(response.text[:500])

with open("manhattan_subway_jan_may_2026.csv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("manhattan_subway_jan_may_2026.csv", low_memory=False)

print(df.head())
print(df.columns)
print(df.shape)

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])

print("Downloaded rows:", len(df))