"""
train_gat.py
------------
Training script for the ST-GAT-GRU hotspot classifier.

Data layout (one "sample" = one day, all 123 stations processed together):
  x_ride    : (N, SEQ_LEN)  — 7-day scaled ridership window per station
  x_cafe    : (N, 1)        — static café density per station
  x_weather : (2,)          — today's (temp, precip) shared across stations
  y         : (N,)          — binary hotspot label per station

Split (by DATE — no random splits):
  train : all dates except last 25
  val   : dates[-25:-15]  (10 days)
  test  : dates[-15:]     (15 days)

Outputs:
  outputs/best_gat_model.pt
  outputs/gat_results.json
"""

import sys, os, json, math, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
)

from gat_model import SimplifiedGATModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SUBWAY_CSV  = "../data/subway_data_2026.csv"
CAFE_CSV    = "../data/manhattan_cafes.csv"
META_CSV    = "../data/station_metadata.csv"
ADJ_PT      = "../data/adjacency_matrix.pt"
WEATHER_CSV = "../data/weather_2026_daily.csv"   # preprocessed by prep_2026_data.py
OUTPUT_DIR  = "../outputs"
CKPT_PATH   = os.path.join(OUTPUT_DIR, "best_gat_model.pt")
RESULTS_PATH= os.path.join(OUTPUT_DIR, "gat_results.json")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEQ_LEN    = 7
HIDDEN_DIM = 64
EMBED_DIM  = 64
GAT_HEADS  = 4
DROPOUT    = 0.3
EPOCHS     = 50
LR         = 1e-2
WD         = 1e-3
THRESHOLD  = 0.3
SEED       = 42

EARTH_RADIUS_M = 6_371_000
CAFE_RADIUS_M  = 400

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_station_order():
    """Return ordered list of station_complex_id strings matching adj matrix."""
    meta = pd.read_csv(META_CSV)
    return meta["station_id"].astype(str).tolist()   # 123 ids, adj-matrix order


def load_subway(station_ids):
    """
    Returns:
      ride_mat  : np.ndarray (D, N) — daily morning ridership (scaled later)
      hot_mat   : np.ndarray (D, N) — binary hotspot labels
      valid_mat : np.ndarray (D, N) — 1 where station-day exists in raw data
      dates     : list of D pd.Timestamp
    """
    df = pd.read_csv(SUBWAY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["station_complex_id"] = df["station_complex_id"].astype(str)

    # Recompute hotspot globally: top 25% ridership across all station-days
    threshold = df["morning_ridership"].quantile(0.75)
    df["hotspot"] = (df["morning_ridership"] >= threshold).astype(int)

    dates = sorted(df["date"].unique())

    # Pivot to (D × N) matrices; NaN = missing station-day
    ride_piv = df.pivot_table(
        index="date", columns="station_complex_id",
        values="morning_ridership"
    ).reindex(index=dates, columns=station_ids)

    hot_piv = df.pivot_table(
        index="date", columns="station_complex_id",
        values="hotspot"
    ).reindex(index=dates, columns=station_ids)

    valid_mat = (~ride_piv.isna()).to_numpy(dtype=np.float32)  # (D, N)
    ride_mat  = ride_piv.fillna(0.0).to_numpy(dtype=np.float32)
    hot_mat   = hot_piv.fillna(0.0).to_numpy(dtype=np.float32)

    return ride_mat, hot_mat, valid_mat, dates


def load_cafe_density(station_ids):
    """Compute café count within 400m for each station (in adj-matrix order)."""
    meta  = pd.read_csv(META_CSV)
    cafe  = pd.read_csv(CAFE_CSV).dropna(subset=["latitude", "longitude"])

    station_coords = meta[["lat", "lon"]].to_numpy()
    cafe_coords    = cafe[["latitude", "longitude"]].to_numpy()

    tree   = BallTree(np.radians(cafe_coords), metric="haversine")
    radius = CAFE_RADIUS_M / EARTH_RADIUS_M
    counts = tree.query_radius(np.radians(station_coords), r=radius, count_only=True)

    return counts.astype(np.float32)   # (N,)


def load_weather(dates):
    """
    Load preprocessed daily weather from weather_2026_daily.csv
    (produced by prep_2026_data.py — Central Park station USW00094728).
    Returns (D, 2) array: [tmax_c, prcp_mm] per day.
    """
    wx = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    wx["date"] = wx["date"].dt.date
    wx = wx.set_index("date").reindex(dates, fill_value=0.0)

    temp   = wx["tmax_c"].to_numpy(dtype=np.float32)
    precip = wx["prcp_mm"].to_numpy(dtype=np.float32)

    n_real = (temp != 0).sum()
    print(f"  [weather] {n_real}/{len(dates)} days with real temp data  "
          f"(range {temp.min():.1f}°C – {temp.max():.1f}°C)")
    return np.stack([temp, precip], axis=1)   # (D, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. GRAPH UTILS
# ═════════════════════════════════════════════════════════════════════════════

def adj_to_edge_index(adj):
    """Convert (N, N) adjacency tensor to edge_index (2, E) and edge_weight (E,)."""
    ei = adj.nonzero(as_tuple=False).t().contiguous()  # (2, E)
    ew = adj[ei[0], ei[1]]                              # (E,)
    return ei, ew


# ═════════════════════════════════════════════════════════════════════════════
# 3. SEQUENCE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_day_tensors(t, ride_scaled, cafe_density, weather_mat, hot_mat, device):
    """
    Build SimplifiedGATModel inputs for date index t (target day).
    t must be >= SEQ_LEN.

    Returns tensors on `device`:
      x_ride    : (N, SEQ_LEN)  — flattened ridership window
      x_cafe    : (N, 1)
      x_weather : (2,)          — today's weather, shared across stations
      y         : (N,)
    """
    ride_window = ride_scaled[t - SEQ_LEN:t, :]              # (SEQ_LEN, N)
    x_ride    = torch.tensor(ride_window.T, dtype=torch.float32, device=device)  # (N, SEQ_LEN)
    x_cafe    = torch.tensor(cafe_density,  dtype=torch.float32, device=device).unsqueeze(-1)
    x_weather = torch.tensor(weather_mat[t - 1], dtype=torch.float32, device=device)
    y         = torch.tensor(hot_mat[t], dtype=torch.float32, device=device)
    return x_ride, x_cafe, x_weather, y


# ═════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION HELPER
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model, indices, ride_scaled, cafe_density, weather_mat, hot_mat,
             valid_mat, edge_index, edge_weight, device, print_mean_prob=False):
    """Run model over a list of date indices; return metrics dict.
    Model forward() returns logits; sigmoid applied here for probs.
    """
    model.eval()
    all_logits, all_true, all_valid = [], [], []

    with torch.no_grad():
        for t in indices:
            x_ride, x_cafe, x_weather, y = build_day_tensors(
                t, ride_scaled, cafe_density, weather_mat, hot_mat, device)
            logits = model(x_ride, x_cafe, x_weather, edge_index, edge_weight)
            all_logits.append(logits.cpu().numpy())
            all_true.append(y.cpu().numpy())
            all_valid.append(valid_mat[t])

    logits_arr = np.concatenate(all_logits)
    probs_arr  = 1.0 / (1.0 + np.exp(-logits_arr))   # sigmoid
    true_arr   = np.concatenate(all_true)
    valid_arr  = np.concatenate(all_valid).astype(bool)

    if print_mean_prob:
        print(f"  [diag] mean predicted prob (all stations, val days): {probs_arr[valid_arr].mean():.4f}")
        print(f"  [diag] prob percentiles [10,50,90]: "
              f"{np.percentile(probs_arr[valid_arr],[10,50,90])}")

    # Evaluate only on station-days that exist in the raw data
    probs_arr = probs_arr[valid_arr]
    true_arr  = true_arr[valid_arr].astype(int)
    preds_arr = (probs_arr >= THRESHOLD).astype(int)

    metrics = {
        "f1":        float(f1_score(true_arr, preds_arr, zero_division=0)),
        "auc":       float(roc_auc_score(true_arr, probs_arr))
                     if len(np.unique(true_arr)) > 1 else float("nan"),
        "precision": float(precision_score(true_arr, preds_arr, zero_division=0)),
        "recall":    float(recall_score(true_arr, preds_arr, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_arr, preds_arr).tolist(),
    }
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    station_ids  = load_station_order()
    N            = len(station_ids)

    ride_mat, hot_mat, valid_mat, dates = load_subway(station_ids)
    D = len(dates)
    print(f"  Stations : {N}")
    print(f"  Dates    : {D}  ({dates[0].date()} → {dates[-1].date()})")

    cafe_density = load_cafe_density(station_ids)
    print(f"  Café density range: [{cafe_density.min():.0f}, {cafe_density.max():.0f}]")

    weather_mat = load_weather(dates)

    # ── Date split ───────────────────────────────────────────────────────────
    test_dates  = dates[-15:]
    val_dates   = dates[-25:-15]
    train_dates = dates[:-25]

    print(f"\nDate splits:")
    print(f"  Train : {train_dates[0].date()} → {train_dates[-1].date()}  ({len(train_dates)} days)")
    print(f"  Val   : {val_dates[0].date()}   → {val_dates[-1].date()}    ({len(val_dates)} days)")
    print(f"  Test  : {test_dates[0].date()}  → {test_dates[-1].date()}   ({len(test_dates)} days)")

    # Indices into the (D, N) matrices
    date_to_idx = {d: i for i, d in enumerate(dates)}
    train_idx = [date_to_idx[d] for d in train_dates if date_to_idx[d] >= SEQ_LEN]
    val_idx   = [date_to_idx[d] for d in val_dates   if date_to_idx[d] >= SEQ_LEN]
    test_idx  = [date_to_idx[d] for d in test_dates  if date_to_idx[d] >= SEQ_LEN]
    print(f"  Usable train days (after {SEQ_LEN}-day lookback): {len(train_idx)}")

    # ── Scale ridership on train data only ────────────────────────────────────
    train_ride = ride_mat[train_idx, :]           # (train_days, N)
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
    print(f"\nGraph: {N} nodes, {edge_index.shape[1]} directed edges")

    # ── Ridership feature diagnostic ──────────────────────────────────────────
    train_windows = np.stack([ride_scaled[t - SEQ_LEN:t, :] for t in train_idx])
    print(f"\nRidership feature check (train windows):")
    print(f"  global  mean={train_windows.mean():.4f}  std={train_windows.std():.4f}")
    per_st_std = train_windows.reshape(-1, N).std(axis=0)
    print(f"  per-station std: mean={per_st_std.mean():.4f}  "
          f"min={per_st_std.min():.4f}  max={per_st_std.max():.4f}")

    # ── Model (simplified: no outer GRU) ──────────────────────────────────────
    model = SimplifiedGATModel(
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        gat_heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SimplifiedGATModel  params={total_params:,}  "
          f"(no outer temporal GRU — diagnosing convergence)")

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels = hot_mat[train_idx].flatten()
    train_valid  = valid_mat[train_idx].flatten().astype(bool)
    n_pos = train_labels[train_valid].sum()
    n_neg = train_valid.sum() - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Pos-weight (neg/pos): {pos_weight:.2f}  (pos={n_pos:.0f}, neg={n_neg:.0f})")

    # ── Graph sanity check ────────────────────────────────────────────────────
    print(f"\n[diag] edge_index shape : {edge_index.shape}  (expect (2, 3450))")
    print(f"[diag] edge_weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")

    # ── Loss: BCEWithLogitsLoss with pos_weight tensor ────────────────────────
    # Model forward() returns logits. BCEWithLogitsLoss applies sigmoid
    # internally using the numerically stable log-sum-exp trick.
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor, reduction="none"
    )

    def masked_loss(logits, targets, valid_mask):
        """BCEWithLogitsLoss applied only to valid station-days."""
        loss_all = criterion(logits, targets) * valid_mask
        return loss_all.sum() / valid_mask.sum().clamp(min=1)

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1   = -1.0
    best_epoch    = -1
    first_step_done = False   # flag for one-shot diagnostics

    print(f"\n{'─'*60}")
    print(f"Training ST-GAT-GRU  |  lr={LR}  threshold={THRESHOLD}  epochs={EPOCHS}")
    print(f"{'─'*60}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for t in train_idx:
            x_ride, x_cafe, x_weather, y = build_day_tensors(
                t, ride_scaled, cafe_density_scaled, weather_scaled, hot_mat, device)
            valid_mask = torch.tensor(valid_mat[t], dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # On the very first step: run with debug prints to verify graph
            _dbg = (not first_step_done)
            logits = model(x_ride, x_cafe, x_weather,
                           edge_index, edge_weight, _debug=_dbg)

            if _dbg:
                probs_dbg = torch.sigmoid(logits)
                print(f"  [diag] first step —")
                print(f"    logits: min={logits.min():.4f} max={logits.max():.4f} "
                      f"std={logits.std():.4f}")
                print(f"    probs : mean={probs_dbg.mean():.4f}  "
                      f"unique values: {len(logits.unique())}")
                print(f"    valid_mask sum: {valid_mask.sum():.0f} / {len(valid_mask)}")
                print(f"    pos labels in this day: {y.sum():.0f}")

            loss = masked_loss(logits, y, valid_mask)
            loss.backward()

            if _dbg:
                # Check gradient norms IMMEDIATELY after backward, before zero_grad
                print(f"  [diag] gradient norms (after first backward):")
                for name, module in [
                    ("ride_enc",     model.ride_enc),
                    ("cafe_enc",     model.cafe_enc),
                    ("weather_enc",  model.weather_enc),
                    ("fusion",       model.fusion),
                    ("gat1",         model.gat1),
                    ("gat2",         model.gat2),
                    ("head",         model.head),
                ]:
                    norms = [p.grad.norm().item() for p in module.parameters()
                             if p.grad is not None]
                    avg_n = sum(norms) / len(norms) if norms else 0.0
                    flag  = "" if avg_n > 1e-8 else "  ← DEAD"
                    print(f"    {name:<14}: {avg_n:.5f}{flag}")
                first_step_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_idx)

        diag_epoch = epoch in (1, 10, 20)
        val_metrics = evaluate(
            model, val_idx, ride_scaled, cafe_density_scaled,
            weather_scaled, hot_mat, valid_mat, edge_index, edge_weight,
            device, print_mean_prob=diag_epoch)

        # Step scheduler on val AUC
        scheduler.step(val_metrics["auc"])
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | "
                  f"val_F1={val_metrics['f1']:.4f} | val_AUC={val_metrics['auc']:.4f} | "
                  f"lr={current_lr:.2e}")

        if diag_epoch and epoch > 1:
            # Re-run one backward to get fresh gradient norms at epoch 10/20
            model.train()
            t0 = train_idx[0]
            x_r, x_c, x_w, y0 = build_day_tensors(
                t0, ride_scaled, cafe_density_scaled, weather_scaled, hot_mat, device)
            vm = torch.tensor(valid_mat[t0], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            l0 = masked_loss(model(x_r, x_c, x_w, edge_index, edge_weight), y0, vm)
            l0.backward()
            print(f"  [diag epoch {epoch}] gradient norms:")
            for name, module in [
                ("ride_enc", model.ride_enc), ("cafe_enc",  model.cafe_enc),
                ("weather_enc", model.weather_enc),
                ("fusion",  model.fusion),    ("gat1",     model.gat1),
                ("gat2",    model.gat2),      ("head",     model.head),
            ]:
                norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
                avg_n = sum(norms)/len(norms) if norms else 0.0
                flag  = "" if avg_n > 1e-6 else "  ← DEAD"
                print(f"    {name:<14}: {avg_n:.5f}{flag}")
            optimizer.zero_grad()

        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT_PATH)

    print(f"\nBest val F1={best_val_f1:.4f} at epoch {best_epoch} → {CKPT_PATH}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint and evaluating on test set...")
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))

    test_metrics = evaluate(
        model, test_idx, ride_scaled, cafe_density_scaled,
        weather_scaled, hot_mat, valid_mat, edge_index, edge_weight, device)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model": "SimplifiedGAT (no outer GRU)",
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "test": {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in test_metrics.items()},
        "hyperparameters": {
            "seq_len": SEQ_LEN, "hidden_dim": HIDDEN_DIM,
            "embed_dim": EMBED_DIM, "gat_heads": GAT_HEADS,
            "dropout": DROPOUT, "epochs": EPOCHS, "lr": LR, "weight_decay": WD,
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {RESULTS_PATH}")

    # ── Comparison table ──────────────────────────────────────────────────────
    f1  = test_metrics["f1"]
    auc = test_metrics["auc"]

    print(f"\n{'═'*52}")
    print(f"  Model comparison")
    print(f"{'─'*52}")
    print(f"  {'Model':<28} {'F1':>6}  {'AUC':>6}")
    print(f"{'─'*52}")
    print(f"  {'Model 0 — LSTM (ride only)':<28} {'0.48':>6}  {'0.71':>6}")
    print(f"  {'Model 1 — LSTM + café':<28} {'0.59':>6}  {'0.79':>6}")
    print(f"  {'ST-GAT-GRU (full)':<28} {f1:>6.4f}  {auc:>6.4f}")
    print(f"{'═'*52}")

    cm = test_metrics["confusion_matrix"]
    print(f"\nConfusion matrix (test):")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\nPrecision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
