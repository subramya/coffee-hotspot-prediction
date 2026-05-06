"""
make_figures.py
---------------
Generate all 5 blog figures and save to ../assets/
Run from the script/ directory.
"""

import os, sys, json, shutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Global style ──────────────────────────────────────────────────────────────
BG      = "#1a0f0a"
TEXT    = "#e8d9c4"
ACCENT  = "#d4924a"
MUTED   = "#7a4f2e"
SUBTEXT = "#a08060"
CREAM   = "#f0e6d3"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "axes.edgecolor":    SUBTEXT,
    "grid.color":        SUBTEXT,
    "grid.alpha":        0.2,
    "font.family":       "DejaVu Sans",
})

ASSETS = "../assets"
os.makedirs(ASSETS, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
META_CSV    = "../data/station_metadata.csv"
SUBWAY_CSV  = "../data/subway_data_2026.csv"
ADJ_PT      = "../data/adjacency_matrix.pt"
CCI_CSV     = "../data/cci_scores.csv"
GAT_JSON    = "../outputs/gat_results.json"
CCI_PLOT    = "../outputs/cci_barplot.png"


# ═════════════════════════════════════════════════════════════════════════════
# Shared data loading
# ═════════════════════════════════════════════════════════════════════════════

meta = pd.read_csv(META_CSV)
meta["station_id"] = meta["station_id"].astype(str)

df26 = pd.read_csv(SUBWAY_CSV)
df26["date"] = pd.to_datetime(df26["date"])
df26["station_complex_id"] = df26["station_complex_id"].astype(str)
THRESHOLD = df26["morning_ridership"].quantile(0.75)  # 3258
df26["hotspot"] = (df26["morning_ridership"] >= THRESHOLD).astype(int)

with open(GAT_JSON) as f:
    gat = json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Manhattan hotspot frequency map
# ═════════════════════════════════════════════════════════════════════════════

def fig1():
    # Hotspot frequency per station
    freq = (df26.groupby("station_complex_id")["hotspot"]
                .mean().reset_index()
                .rename(columns={"hotspot": "freq"}))
    freq["station_complex_id"] = freq["station_complex_id"].astype(str)
    stations = meta.merge(freq, left_on="station_id", right_on="station_complex_id", how="left")
    stations["freq"] = stations["freq"].fillna(0)

    # Top 5 by frequency
    top5 = stations.nlargest(5, "freq")

    # Colormap: muted brown → gold
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hotspot_warm", ["#3d1a0a", "#8b4513", ACCENT, "#f5c842"], N=256)

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    sc = ax.scatter(
        stations["lon"], stations["lat"],
        c=stations["freq"], cmap=cmap, vmin=0, vmax=1,
        s=55, zorder=3, edgecolors="none", alpha=0.92
    )

    # Label top 5
    already_nudged = {}
    nudges = {
        "611": (-0.018, 0.003),   # Times Sq
        "610": ( 0.013, 0.003),   # Grand Central
        "164": (-0.016,-0.004),   # Penn A,C,E
        "607": ( 0.013,-0.004),   # Herald Sq
        "318": (-0.018,-0.010),   # Penn 1,2,3
    }
    for _, row in top5.iterrows():
        nx, ny = nudges.get(str(row["station_id"]), (0.010, 0.005))
        ax.annotate(
            row["station_name"].split("(")[0].strip(),
            xy=(row["lon"], row["lat"]),
            xytext=(row["lon"] + nx, row["lat"] + ny),
            fontsize=7.5, color=CREAM,
            arrowprops=dict(arrowstyle="-", color=SUBTEXT, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=SUBTEXT, alpha=0.7, lw=0.5),
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Hotspot Frequency", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)
    cbar.outline.set_edgecolor(SUBTEXT)

    ax.set_title("Morning Hotspot Frequency — Manhattan Subway Stations",
                 fontsize=12, color=TEXT, pad=12, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SUBTEXT)

    fig.tight_layout(pad=1.5)
    path = os.path.join(ASSETS, "fig1.png")
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Ridership histogram + station graph
# ═════════════════════════════════════════════════════════════════════════════

def fig2():
    adj = torch.load(ADJ_PT, weights_only=True)
    ei  = adj.nonzero(as_tuple=False).numpy()  # (E, 2)

    # Station positions in adjacency-matrix order (sorted by station_id)
    lats = meta["lat"].to_numpy()
    lons = meta["lon"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # ── Left: ridership histogram ─────────────────────────────────────────────
    ridership = df26["morning_ridership"].clip(upper=20000)  # clip extreme tail for readability
    ax1.hist(ridership, bins=60, color=ACCENT, alpha=0.85, edgecolor="none")
    ax1.axvline(THRESHOLD, color=CREAM, linewidth=1.8, linestyle="--")
    ax1.text(THRESHOLD + 200, ax1.get_ylim()[1] * 0.88,
             f"hotspot threshold\n({int(THRESHOLD):,} riders)",
             color=CREAM, fontsize=8.5, va="top")
    ax1.set_xlabel("Morning Ridership (7–10 am)", fontsize=10)
    ax1.set_ylabel("Station-days", fontsize=10)
    ax1.set_title("Ridership Distribution", fontsize=11, color=TEXT, fontweight="bold")
    ax1.tick_params(labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax1.spines[sp].set_edgecolor(SUBTEXT)

    # ── Right: station graph ──────────────────────────────────────────────────
    # Draw edges first (thin, low alpha)
    for i, j in ei:
        if i < j:   # undirected — draw each edge once
            ax2.plot([lons[i], lons[j]], [lats[i], lats[j]],
                     color=SUBTEXT, lw=0.4, alpha=0.3, zorder=1)

    ax2.scatter(lons, lats, s=22, color=ACCENT, zorder=3, edgecolors="none", alpha=0.9)
    ax2.set_title("Station Graph (proximity + shared line)", fontsize=11, color=TEXT, fontweight="bold")
    ax2.set_xlabel("Longitude", fontsize=9)
    ax2.set_ylabel("Latitude", fontsize=9)
    ax2.tick_params(labelsize=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor(SUBTEXT)

    fig.suptitle("Data Overview", fontsize=13, color=TEXT, fontweight="bold", y=1.01)
    fig.tight_layout(pad=1.5)
    path = os.path.join(ASSETS, "fig2.png")
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Architecture diagram (matplotlib only)
# ═════════════════════════════════════════════════════════════════════════════

def fig3():
    fig, ax = plt.subplots(figsize=(9, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    BOX_W   = 2.6
    BOX_H   = 0.85
    BOX_CLR = ACCENT
    TXT_CLR = BG
    LBL_CLR = TEXT
    ARR_CLR = CREAM

    def box(cx, cy, label, sublabel=None, w=BOX_W, color=BOX_CLR):
        patch = FancyBboxPatch(
            (cx - w/2, cy - BOX_H/2), w, BOX_H,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=ARR_CLR, linewidth=1.2, zorder=3
        )
        ax.add_patch(patch)
        y_text = cy + (0.12 if sublabel else 0)
        ax.text(cx, y_text, label, ha="center", va="center",
                fontsize=9.5, color=TXT_CLR, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.18, sublabel, ha="center", va="center",
                    fontsize=7.5, color=TXT_CLR, alpha=0.85, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2 + BOX_H/2), xytext=(x1, y1 - BOX_H/2),
            arrowprops=dict(
                arrowstyle="-|>", color=ARR_CLR, lw=1.4,
                mutation_scale=12
            ), zorder=2
        )

    def fan_arrow(sources_xy, target_cx, target_cy):
        """Draw arrows from multiple sources to one target box."""
        for sx, sy in sources_xy:
            ax.annotate(
                "", xy=(target_cx, target_cy + BOX_H/2),
                xytext=(sx, sy - BOX_H/2),
                arrowprops=dict(
                    arrowstyle="-|>", color=ARR_CLR, lw=1.2,
                    connectionstyle="arc3,rad=0.0",
                    mutation_scale=10
                ), zorder=2
            )

    # ── Input branch boxes (row 1) ────────────────────────────────────────────
    branch_y = 12.5
    branches = [
        (2.0, "Ridership Branch", "7-day window → Linear+ReLU"),
        (5.0, "Café Branch",      "count within 400m → Linear+ReLU"),
        (8.0, "Weather Branch",   "temp + precip → Linear+ReLU"),
    ]
    for cx, lbl, sub in branches:
        box(cx, branch_y, lbl, sub)

    # Input data labels above branches
    input_labels = ["Ridership\n(N × 7)", "Café Density\n(N × 1)", "Weather\n(2,)"]
    for (cx, _, _), inp in zip(branches, input_labels):
        ax.text(cx, branch_y + BOX_H/2 + 0.45, inp,
                ha="center", va="bottom", fontsize=8, color=LBL_CLR, alpha=0.8)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fuse_y = 10.2
    fan_arrow([(cx, branch_y) for cx, _, _ in branches], 5.0, fuse_y)
    box(5.0, fuse_y, "Fusion Layer", "concat 3 branches → Linear+ReLU  (N × 64)", w=4.8)

    # ── GAT 1 ─────────────────────────────────────────────────────────────────
    gat1_y = 8.2
    arrow(5.0, fuse_y, 5.0, gat1_y)
    box(5.0, gat1_y, "GAT Layer 1",
        "GATConv(64→64, heads=4, concat=True)  →  (N × 256)", w=5.2)

    # ── GAT 2 + residual ──────────────────────────────────────────────────────
    gat2_y = 6.2
    arrow(5.0, gat1_y, 5.0, gat2_y)
    box(5.0, gat2_y, "GAT Layer 2 + Residual",
        "GATConv(256→64, heads=4, concat=False)  +  skip  →  (N × 64)", w=5.8)

    # Residual skip arrow (curved, from fuse to gat2)
    ax.annotate(
        "", xy=(7.8, gat2_y + BOX_H/2),
        xytext=(7.8, fuse_y - BOX_H/2),
        arrowprops=dict(
            arrowstyle="-|>", color=MUTED, lw=1.0, linestyle="dashed",
            connectionstyle="arc3,rad=-0.4", mutation_scale=10
        ), zorder=2
    )
    ax.text(8.5, (fuse_y + gat2_y) / 2, "residual",
            fontsize=7.5, color=MUTED, va="center", ha="left", alpha=0.85)

    # ── Classification head ───────────────────────────────────────────────────
    head_y = 4.2
    arrow(5.0, gat2_y, 5.0, head_y)
    box(5.0, head_y, "Classification Head",
        "Dropout(0.3) → Linear(64→1)  →  logit per station", w=5.0)

    # ── Output ────────────────────────────────────────────────────────────────
    out_y = 2.3
    arrow(5.0, head_y, 5.0, out_y)
    out_patch = FancyBboxPatch(
        (5.0 - 2.0, out_y - 0.5), 4.0, 1.0,
        boxstyle="round,pad=0.15",
        facecolor="#2a1408", edgecolor=ACCENT, linewidth=2.0, zorder=3
    )
    ax.add_patch(out_patch)
    ax.text(5.0, out_y + 0.08, "Hotspot Probability", ha="center", va="center",
            fontsize=10, color=ACCENT, fontweight="bold", zorder=4)
    ax.text(5.0, out_y - 0.22, "sigmoid(logit)  →  p ∈ [0, 1] per station",
            ha="center", va="center", fontsize=7.8, color=TEXT, zorder=4)

    ax.set_title("ST-GAT Model Architecture", fontsize=14, color=TEXT,
                 fontweight="bold", pad=8)

    fig.tight_layout(pad=0.5)
    path = os.path.join(ASSETS, "fig3.png")
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Model comparison bar chart
# ═════════════════════════════════════════════════════════════════════════════

def fig4():
    models = ["LSTM\n(ride only)", "LSTM\n(ride + café)", "ST-GAT\n(full)"]
    f1s    = [0.48, 0.59, gat["test"]["f1"]]
    aucs   = [0.71, 0.79, gat["test"]["auc"]]

    x      = np.arange(len(models))
    width  = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars_f1  = ax.bar(x - width/2, f1s,  width, label="F1",  color=ACCENT, edgecolor="none")
    bars_auc = ax.bar(x + width/2, aucs, width, label="AUC", color=CREAM,  edgecolor="none", alpha=0.85)

    # Value labels on bars
    for bars, vals in [(bars_f1, f1s), (bars_auc, aucs)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison — Test Set Performance",
                 fontsize=12, color=TEXT, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.15, labelcolor=TEXT,
              facecolor=BG, edgecolor=SUBTEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_edgecolor(SUBTEXT)
    ax.tick_params(labelsize=9)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    # Highlight GAT bar group
    ax.axvspan(1.6, 2.4, color=ACCENT, alpha=0.06, zorder=0)
    ax.text(2.0, 1.06, "★", ha="center", fontsize=13, color=ACCENT)

    fig.tight_layout(pad=1.5)
    path = os.path.join(ASSETS, "fig4.png")
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 — CCI diverging barplot (copy existing)
# ═════════════════════════════════════════════════════════════════════════════

def fig5():
    src  = CCI_PLOT
    dest = os.path.join(ASSETS, "fig5.png")
    shutil.copy2(src, dest)
    print(f"  Copied {src} → {dest}")


# ═════════════════════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")
    fig1(); fig2(); fig3(); fig4(); fig5()

    print("\n── File check ──────────────────────────────────────────────────")
    all_ok = True
    for i in range(1, 6):
        p = os.path.join(ASSETS, f"fig{i}.png")
        ok = os.path.exists(p)
        size = os.path.getsize(p) if ok else 0
        status = "OK" if ok else "MISSING"
        print(f"  {status}  fig{i}.png  ({size:,} bytes)")
        if not ok:
            all_ok = False
    print()
    print("All figures saved: fig1.png through fig5.png" if all_ok
          else "ERROR: one or more figures missing")
