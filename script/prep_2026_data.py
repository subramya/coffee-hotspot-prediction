"""
prep_2026_data.py
-----------------
Processes the raw 2026 data files into clean CSVs ready for train_gat.py.

Inputs
------
  data/manhattan_subway_jan_may_2026.csv  — raw MTA hourly ridership (already
      filtered to Manhattan, hours 7-10am)
  data/weather2026.csv                   — NOAA GHCND file (all global stations)

Outputs
-------
  data/subway_data_2026.csv       — same schema as subway_data.csv:
      date, station_complex_id, station_complex, latitude, longitude,
      morning_ridership, hotspot
  data/weather_2026_daily.csv     — aligned daily weather:
      date, tmax_c, prcp_mm
"""

import pandas as pd
import numpy as np

SUBWAY_RAW   = "../data/manhattan_subway_jan_may_2026.csv"
WEATHER_RAW  = "../data/weather2026.csv"
SUBWAY_OUT   = "../data/subway_data_2026.csv"
WEATHER_OUT  = "../data/weather_2026_daily.csv"

NYC_STATION  = "USW00094728"   # Central Park, NY


# ── 1. Subway ─────────────────────────────────────────────────────────────────

def process_subway():
    print("Processing subway data...")
    df = pd.read_csv(SUBWAY_RAW, low_memory=False)

    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["date"] = df["transit_timestamp"].dt.date

    # Already filtered to Manhattan + hours 7-10; aggregate to daily per station
    agg = df.groupby(
        ["date", "station_complex_id", "station_complex", "latitude", "longitude"],
        as_index=False
    )["ridership"].sum()
    agg = agg.rename(columns={"ridership": "morning_ridership"})

    # Hotspot: top 25% ridership across ALL station-days
    threshold = agg["morning_ridership"].quantile(0.75)
    agg["hotspot"] = (agg["morning_ridership"] >= threshold).astype(int)

    agg = agg.sort_values(["date", "station_complex_id"]).reset_index(drop=True)
    agg.to_csv(SUBWAY_OUT, index=False)

    print(f"  Saved: {SUBWAY_OUT}")
    print(f"  Rows            : {len(agg)}")
    print(f"  Unique dates    : {agg['date'].nunique()}  "
          f"({agg['date'].min()} → {agg['date'].max()})")
    print(f"  Unique stations : {agg['station_complex_id'].nunique()}")
    print(f"  Hotspot rate    : {agg['hotspot'].mean():.2%}")
    return agg


# ── 2. Weather ────────────────────────────────────────────────────────────────

def process_weather(subway_dates):
    print("\nProcessing weather data (Central Park, USW00094728)...")

    # Build the set of dates we need in YYYYMMDD format
    date_strs = {str(d).replace("-", "") for d in subway_dates}

    # Stream through the large GHCND file, keeping only what we need
    chunks = pd.read_csv(
        WEATHER_RAW,
        header=None,
        names=["station", "date", "element", "value",
               "m_flag", "q_flag", "s_flag", "obs_time"],
        dtype={"date": str, "value": float},
        usecols=["station", "date", "element", "value"],
        on_bad_lines="skip",
        chunksize=500_000,
    )

    rows = []
    for chunk in chunks:
        sub = chunk[
            (chunk["station"] == NYC_STATION) &
            (chunk["element"].isin(["TMAX", "PRCP"])) &
            (chunk["date"].isin(date_strs))
        ]
        rows.append(sub)

    wx = pd.concat(rows, ignore_index=True)
    print(f"  Rows extracted  : {len(wx)}")

    if wx.empty:
        print("  WARNING: no weather rows found — writing zeros")
        out = pd.DataFrame({"date": subway_dates, "tmax_c": 0.0, "prcp_mm": 0.0})
        out.to_csv(WEATHER_OUT, index=False)
        return out

    # Pivot to one row per date
    wx["date_parsed"] = pd.to_datetime(wx["date"], format="%Y%m%d").dt.date
    piv = wx.pivot_table(index="date_parsed", columns="element",
                         values="value", aggfunc="mean")
    piv = piv.reindex(subway_dates)

    # GHCND units: TMAX in tenths of °C, PRCP in tenths of mm
    out = pd.DataFrame({
        "date":    subway_dates,
        "tmax_c":  (piv.get("TMAX", pd.Series(np.nan, index=subway_dates)) / 10.0).values,
        "prcp_mm": (piv.get("PRCP", pd.Series(0.0,   index=subway_dates)) / 10.0).fillna(0.0).values,
    })
    # Fill any missing temp with forward-fill then mean
    out["tmax_c"] = out["tmax_c"].ffill().bfill().fillna(out["tmax_c"].mean())

    out.to_csv(WEATHER_OUT, index=False)
    print(f"  Saved: {WEATHER_OUT}")
    print(f"  Date range: {out['date'].min()} → {out['date'].max()}")
    print(f"  Temp range: {out['tmax_c'].min():.1f}°C – {out['tmax_c'].max():.1f}°C")
    print(f"  Precip range: {out['prcp_mm'].min():.1f} – {out['prcp_mm'].max():.1f} mm")
    print(f"  Missing temps filled: {out['tmax_c'].isna().sum()}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subway_df = process_subway()
    subway_dates = sorted(subway_df["date"].unique())
    process_weather(subway_dates)
    print("\nDone. Ready for train_gat.py.")
