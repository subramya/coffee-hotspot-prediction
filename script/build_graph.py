"""
build_graph.py
--------------
Builds a station graph for the ST-GAT-GRU model.

Nodes  : 123 Manhattan subway stations
Edges  : stations within 1 km of each other OR sharing at least one line
Weights: inverse_distance (capped) + line_sharing_bonus, normalised to [0,1]

Outputs
-------
data/adjacency_matrix.pt    – torch.FloatTensor, shape (123, 123)
data/station_metadata.csv   – station_id, station_name, lat, lon, num_edges
"""

import math
import re
import pandas as pd
import torch

# ── paths ────────────────────────────────────────────────────────────────────
SUBWAY_CSV  = "../data/subway_data.csv"
ADJ_OUT     = "../data/adjacency_matrix.pt"
META_OUT    = "../data/station_metadata.csv"

# ── hyper-parameters ─────────────────────────────────────────────────────────
PROXIMITY_KM      = 1.0   # edge if distance <= this
LINE_BONUS        = 0.5   # added to weight when stations share a line
MIN_DIST_KM       = 0.05  # floor to avoid division by zero for co-located stations

# ── haversine ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

# ── line parsing ─────────────────────────────────────────────────────────────
def parse_lines(station_complex_name):
    """Extract set of line identifiers from e.g. '49 St (N,R,W)'."""
    match = re.search(r'\(([^)]+)\)', station_complex_name)
    if not match:
        return set()
    return {t.strip() for t in match.group(1).split(',')}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(SUBWAY_CSV)

    # one row per station (stable ordering by station_complex_id)
    stations = (
        df.drop_duplicates("station_complex_id")
          .sort_values("station_complex_id")
          .reset_index(drop=True)
    )
    N = len(stations)
    print(f"Stations loaded: {N}")

    lats  = stations["latitude"].tolist()
    lons  = stations["longitude"].tolist()
    names = stations["station_complex"].tolist()
    ids   = stations["station_complex_id"].tolist()

    lines_per_station = [parse_lines(n) for n in names]

    # ── build raw weight matrix ───────────────────────────────────────────────
    W = [[0.0] * N for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            dist_km = haversine(lats[i], lons[i], lats[j], lons[j])
            shared  = bool(lines_per_station[i] & lines_per_station[j])

            connected = (dist_km <= PROXIMITY_KM) or shared
            if not connected:
                continue

            # inverse-distance component (capped at min dist)
            d = max(dist_km, MIN_DIST_KM)
            w = 1.0 / d

            # line-sharing bonus
            if shared:
                w += LINE_BONUS

            W[i][j] = w
            W[j][i] = w

    # ── normalise to [0, 1] ───────────────────────────────────────────────────
    flat_vals = [W[i][j] for i in range(N) for j in range(N) if i != j and W[i][j] > 0]
    if flat_vals:
        w_max = max(flat_vals)
        for i in range(N):
            for j in range(N):
                if W[i][j] > 0:
                    W[i][j] /= w_max

    adj = torch.tensor(W, dtype=torch.float32)

    # ── save adjacency matrix ─────────────────────────────────────────────────
    torch.save(adj, ADJ_OUT)
    print(f"Saved: {ADJ_OUT}  shape={tuple(adj.shape)}")

    # ── station metadata ──────────────────────────────────────────────────────
    num_edges = [(adj[i] > 0).sum().item() for i in range(N)]

    meta = pd.DataFrame({
        "station_id":   ids,
        "station_name": names,
        "lat":          lats,
        "lon":          lons,
        "num_edges":    num_edges,
    })
    meta.to_csv(META_OUT, index=False)
    print(f"Saved: {META_OUT}")

    # ── summary ───────────────────────────────────────────────────────────────
    total_edges = int(adj.gt(0).sum().item()) // 2  # undirected
    avg_degree  = sum(num_edges) / N

    print("\n── Graph Summary ──────────────────────────────")
    print(f"  Nodes      : {N}")
    print(f"  Edges      : {total_edges}")
    print(f"  Avg degree : {avg_degree:.2f}")
    print(f"  Min degree : {min(num_edges)}")
    print(f"  Max degree : {max(num_edges)}")
    print(f"  Density    : {total_edges / (N*(N-1)/2):.3f}")
    print("───────────────────────────────────────────────")

    # sanity: a few high-degree stations
    meta_sorted = meta.sort_values("num_edges", ascending=False).head(5)
    print("\nTop 5 most-connected stations:")
    print(meta_sorted[["station_name", "num_edges"]].to_string(index=False))

if __name__ == "__main__":
    main()
