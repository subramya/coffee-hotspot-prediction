"""
compute_cci.py
--------------
Coffee Contribution Index (CCI): how much does café density around a station
shift the model's predicted hotspot probability?

Method: zero-ablation
  CCI_raw = mean over test days of (baseline_prob − café_zeroed_prob)

  Positive CCI: café density boosts hotspot prediction (high-density downtown).
  Negative CCI: low café density suppresses hotspot prediction (uptown); removing
                café info causes those stations to be over-predicted as hotspots.

The barplot shows a diverging chart — both directions tell the story.

Outputs
-------
  data/cci_scores.csv     — station_id, station_name, latitude, longitude,
                            cci_raw (signed Δprob), cci_score (normalized 0–1)
  outputs/cci_barplot.png — diverging bar chart: top 20 positive + top 15 negative
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from gat_model import SimplifiedGATModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SUBWAY_CSV  = "../data/subway_data_2026.csv"
CAFE_CSV    = "../data/manhattan_cafes.csv"
META_CSV    = "../data/station_metadata.csv"
ADJ_PT      = "../data/adjacency_matrix.pt"
WEATHER_CSV = "../data/weather_2026_daily.csv"
CKPT_PATH   = "../outputs/best_gat_model.pt"
CCI_OUT     = "../data/cci_scores.csv"
PLOT_OUT    = "../outputs/cci_barplot.png"

# ── Hyper-parameters (must match train_gat.py) ────────────────────────────────
SEQ_LEN    = 7
HIDDEN_DIM = 64
EMBED_DIM  = 64
GAT_HEADS  = 4
DROPOUT    = 0.3

EARTH_RADIUS_M = 6_371_000
CAFE_RADIUS_M  = 400


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors train_gat.py exactly so scalers match)
# ═════════════════════════════════════════════════════════════════════════════

def load_station_order():
    meta = pd.read_csv(META_CSV)
    return meta["station_id"].astype(str).tolist()


def load_subway(station_ids):
    df = pd.read_csv(SUBWAY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["station_complex_id"] = df["station_complex_id"].astype(str)

    threshold = df["morning_ridership"].quantile(0.75)
    df["hotspot"] = (df["morning_ridership"] >= threshold).astype(int)

    dates = sorted(df["date"].unique())

    ride_piv = df.pivot_table(
        index="date", columns="station_complex_id",
        values="morning_ridership"
    ).reindex(index=dates, columns=station_ids)

    hot_piv = df.pivot_table(
        index="date", columns="station_complex_id",
        values="hotspot"
    ).reindex(index=dates, columns=station_ids)

    valid_mat = (~ride_piv.isna()).to_numpy(dtype=np.float32)
    ride_mat  = ride_piv.fillna(0.0).to_numpy(dtype=np.float32)
    hot_mat   = hot_piv.fillna(0.0).to_numpy(dtype=np.float32)
    return ride_mat, hot_mat, valid_mat, dates


def load_cafe_density(station_ids):
    meta = pd.read_csv(META_CSV)
    cafe = pd.read_csv(CAFE_CSV).dropna(subset=["latitude", "longitude"])
    station_coords = meta[["lat", "lon"]].to_numpy()
    cafe_coords    = cafe[["latitude", "longitude"]].to_numpy()
    tree   = BallTree(np.radians(cafe_coords), metric="haversine")
    radius = CAFE_RADIUS_M / EARTH_RADIUS_M
    counts = tree.query_radius(np.radians(station_coords), r=radius, count_only=True)
    return counts.astype(np.float32)


def load_weather(dates):
    wx = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    wx["date"] = wx["date"].dt.date
    wx = wx.set_index("date").reindex(dates, fill_value=0.0)
    temp   = wx["tmax_c"].to_numpy(dtype=np.float32)
    precip = wx["prcp_mm"].to_numpy(dtype=np.float32)
    return np.stack([temp, precip], axis=1)


def adj_to_edge_index(adj):
    ei = adj.nonzero(as_tuple=False).t().contiguous()
    ew = adj[ei[0], ei[1]]
    return ei, ew


def build_day_tensors(t, ride_scaled, cafe_density_scaled, weather_scaled, device):
    ride_window = ride_scaled[t - SEQ_LEN:t, :]
    x_ride    = torch.tensor(ride_window.T, dtype=torch.float32, device=device)
    x_cafe    = torch.tensor(cafe_density_scaled, dtype=torch.float32, device=device).unsqueeze(-1)
    x_weather = torch.tensor(weather_scaled[t - 1], dtype=torch.float32, device=device)
    return x_ride, x_cafe, x_weather


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    station_ids  = load_station_order()
    N            = len(station_ids)

    ride_mat, hot_mat, valid_mat, dates = load_subway(station_ids)
    D = len(dates)
    print(f"  Stations : {N}  |  Dates: {D}")

    cafe_density = load_cafe_density(station_ids)
    weather_mat  = load_weather(dates)

    # ── Reproduce exact same splits + scalers as train_gat.py ────────────────
    test_dates  = dates[-15:]
    val_dates   = dates[-25:-15]
    train_dates = dates[:-25]

    date_to_idx = {d: i for i, d in enumerate(dates)}
    train_idx = [date_to_idx[d] for d in train_dates if date_to_idx[d] >= SEQ_LEN]
    test_idx  = [date_to_idx[d] for d in test_dates  if date_to_idx[d] >= SEQ_LEN]
    print(f"  Train days (for scalers): {len(train_idx)}  |  Test days: {len(test_idx)}")

    # Scale ridership (fit on train only)
    train_ride = ride_mat[train_idx, :]
    scaler = StandardScaler()
    scaler.fit(train_ride.reshape(-1, 1))
    ride_scaled = scaler.transform(ride_mat.reshape(-1, 1)).reshape(D, N).astype(np.float32)

    # Scale café density
    cafe_scaler = StandardScaler()
    cafe_density_scaled = cafe_scaler.fit_transform(
        cafe_density.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # Scale weather (fit on train)
    wx_train = weather_mat[train_idx]
    wx_scaler = StandardScaler()
    wx_scaler.fit(wx_train)
    weather_scaled = wx_scaler.transform(weather_mat).astype(np.float32)

    # ── Graph ─────────────────────────────────────────────────────────────────
    adj = torch.load(ADJ_PT, weights_only=True)
    edge_index, edge_weight = adj_to_edge_index(adj)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = SimplifiedGATModel(
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        gat_heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}  |  Eval mode: True")

    # ── CCI computation: zero-ablation ───────────────────────────────────────
    print(f"\nComputing zero-ablation CCI over {len(test_idx)} test days...")

    delta_sum   = np.zeros(N, dtype=np.float64)
    valid_count = np.zeros(N, dtype=np.float64)

    with torch.no_grad():
        for t in test_idx:
            x_ride, x_cafe, x_weather = build_day_tensors(
                t, ride_scaled, cafe_density_scaled, weather_scaled, device)

            baseline_prob = torch.sigmoid(
                model(x_ride, x_cafe, x_weather, edge_index, edge_weight)
            ).cpu().numpy()

            zeroed_prob = model.branch_forward(
                x_ride, x_cafe, x_weather,
                edge_index, edge_weight,
                zero_branches={'cafe'}
            ).cpu().numpy()

            delta = baseline_prob - zeroed_prob
            mask  = valid_mat[t].astype(bool)
            delta_sum[mask]   += delta[mask]
            valid_count[mask] += 1.0

    with np.errstate(invalid="ignore"):
        cci_raw = np.where(valid_count > 0, delta_sum / valid_count, 0.0)

    print(f"  Raw CCI — min={cci_raw.min():.5f}  max={cci_raw.max():.5f}  "
          f"mean={cci_raw.mean():.5f}  std={cci_raw.std():.5f}")
    print(f"  Positive (café boosts): {(cci_raw > 0).sum()}  "
          f"Negative (café suppresses): {(cci_raw < 0).sum()}")

    # Normalize to [0, 1]
    cci_min, cci_max = cci_raw.min(), cci_raw.max()
    if cci_max > cci_min:
        cci_norm = (cci_raw - cci_min) / (cci_max - cci_min)
    else:
        cci_norm = np.zeros_like(cci_raw)

    # ── Build output DataFrame ────────────────────────────────────────────────
    meta = pd.read_csv(META_CSV)
    meta["station_id"] = meta["station_id"].astype(str)

    out = pd.DataFrame({
        "station_id": station_ids,
        "cci_raw":    cci_raw,         # signed Δprob: positive=boost, negative=suppress
        "cci_score":  cci_norm,        # min-max normalized [0,1] for choropleth
    })
    out = out.merge(
        meta[["station_id", "station_name", "lat", "lon"]],
        on="station_id", how="left"
    )
    out = out.rename(columns={"lat": "latitude", "lon": "longitude"})
    out = out[["station_id", "station_name", "latitude", "longitude",
               "cci_raw", "cci_score"]]
    out = out.sort_values("cci_raw", ascending=False).reset_index(drop=True)

    os.makedirs("../data", exist_ok=True)
    out.to_csv(CCI_OUT, index=False)
    print(f"  Saved: {CCI_OUT}")

    # ── Console report ────────────────────────────────────────────────────────
    print("\n── Top 10 (café boosts hotspot prediction) ─────────────────────")
    for i, row in out.head(10).iterrows():
        print(f"  {i+1:2d}.  {row['station_name']:<48}  Δprob={row['cci_raw']:+.5f}")

    print("\n── Bottom 10 (café suppresses hotspot prediction) ──────────────")
    for i, row in out.tail(10).iterrows():
        print(f"  {i+1:3d}.  {row['station_name']:<48}  Δprob={row['cci_raw']:+.5f}")

    # ── Diverging bar chart ───────────────────────────────────────────────────
    # Top 20 positive (café boosts) + top 15 negative (café suppresses)
    top_pos = out[out["cci_raw"] > 0].head(20)
    top_neg = out[out["cci_raw"] < 0].tail(15)
    plot_df = pd.concat([top_neg, top_pos]).reset_index(drop=True)
    # sort ascending so most positive ends up at the top of the chart
    plot_df = plot_df.sort_values("cci_raw").reset_index(drop=True)

    available = {f.name for f in fm.fontManager.ttflist}
    font_name = "Syne" if "Syne" in available else "DejaVu Sans"

    BG      = "#1a0f0a"
    GOLD    = "#d4924a"
    MUTED   = "#7a4f2e"
    TEXT    = "#f0e6d3"
    SUBTEXT = "#a08060"
    ZEROLINE = "#c8a87a"

    bar_colors = [GOLD if v >= 0 else MUTED for v in plot_df["cci_raw"]]

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(
        plot_df["station_name"], plot_df["cci_raw"],
        color=bar_colors, height=0.72, edgecolor="none"
    )

    # Zero reference line
    ax.axvline(0, color=ZEROLINE, linewidth=1.0, alpha=0.7)

    # Value labels
    for bar, val in zip(bars, plot_df["cci_raw"]):
        x_off = 0.0008 if val >= 0 else -0.0008
        ha    = "left"  if val >= 0 else "right"
        ax.text(
            val + x_off, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha=ha, color=TEXT, fontsize=7.5, fontfamily=font_name
        )

    # Annotations marking the two zones
    xmax = plot_df["cci_raw"].abs().max()
    ax.text( xmax * 0.55,  len(plot_df) - 0.3, "café boosts prediction →",
             color=GOLD,  fontsize=8, fontfamily=font_name, alpha=0.85)
    ax.text(-xmax * 0.55,  0.3, "← café suppresses prediction",
             color=MUTED, fontsize=8, fontfamily=font_name, ha="right", alpha=0.85)

    ax.set_xlabel("Δ Predicted Hotspot Probability  (baseline − café-zeroed)",
                  color=TEXT, fontsize=10, fontfamily=font_name, labelpad=10)
    ax.set_title("Coffee Contribution Index — Diverging Attribution",
                 color=TEXT, fontsize=13, fontfamily=font_name,
                 fontweight="bold", pad=14)

    ax.tick_params(colors=TEXT, labelsize=8.5, length=0)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_name); label.set_color(TEXT)
    for label in ax.get_xticklabels():
        label.set_fontfamily(font_name); label.set_color(SUBTEXT)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(SUBTEXT)
    ax.xaxis.grid(True, color=SUBTEXT, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-xmax * 1.22, xmax * 1.22)

    fig.tight_layout(pad=1.5)
    os.makedirs("../outputs", exist_ok=True)
    fig.savefig(PLOT_OUT, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {PLOT_OUT}")

    # ── Confirm both files exist ──────────────────────────────────────────────
    print("\n── Output file check ───────────────────────────────────────────")
    for path in [CCI_OUT, PLOT_OUT]:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        print(f"  {'OK' if exists else 'MISSING':7s}  {path}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
