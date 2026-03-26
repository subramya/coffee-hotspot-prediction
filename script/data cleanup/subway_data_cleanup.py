import pandas as pd

url = "https://data.ny.gov/resource/wujg-7c2s.csv?$limit=200000"
df = pd.read_csv(url, low_memory=False)

df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])

df["hour"] = df["transit_timestamp"].dt.hour
df["date"] = df["transit_timestamp"].dt.date

df = df[df["borough"] == "Manhattan"]
df = df[df["hour"].isin([7, 8, 9, 10])]

final_df = (
    df.groupby(
        ["date", "station_complex_id", "station_complex", "latitude", "longitude"],
        as_index=False
    )["ridership"]
    .sum()
)

final_df = final_df.rename(columns={"ridership": "morning_ridership"})

threshold = final_df["morning_ridership"].quantile(0.75)
final_df["hotspot"] = (final_df["morning_ridership"] >= threshold).astype(int)

final_df.to_csv("subway_data.csv", index=False)

print("Final dataset shape:", final_df.shape)
print(final_df.head())
print("\nHotspot distribution:")
print(final_df["hotspot"].value_counts())