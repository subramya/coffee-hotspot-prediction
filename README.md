# Morning Brews & Morning Rush: Coffee-Driven Subway Hotspots in Manhattan

STAT W3106 – Spring 2026 | Ramya Subramanian, Sachi Patel, Hailey Gamer

**[Blog post →](docs/index.html)**

---

## Overview

We model 7–10am Manhattan subway "hotspots" (top 25% of morning ridership by station-day) as a node-level binary classification problem on a spatial graph. A three-branch Simplified Graph Attention Network (ST-GAT) encodes historical ridership sequences, static café density within 400m of each station, and daily weather, then propagates information across a proximity-and-line graph of 123 Manhattan stations.

We also introduce the **Coffee Contribution Index** (CCI): a zero-ablation attribution measure that quantifies, station by station, how much the café branch shifts the hotspot prediction when removed.

| Model | Features | F1 | AUC |
|-------|----------|----|-----|
| LSTM (Model 0) | Ridership only (3-day) | 0.48 | 0.71 |
| LSTM (Model 1) | Ridership + café density (7-day) | 0.59 | 0.79 |
| **ST-GAT (full)** | All branches + graph | **0.91** | **0.99** |

---

## Repository Structure

```
coffee-hotspot-prediction/
├── docs/
│   └── index.html          # Technical blog post (open in Chrome)
├── notebooks/
│   └── reproduce_figures.ipynb   # Reproduces all 5 blog figures
├── script/
│   ├── gat_model.py        # SimplifiedGATModel definition
│   ├── train_gat.py        # Training loop + evaluation
│   ├── compute_cci.py      # Coffee Contribution Index (zero-ablation)
│   ├── make_figures.py     # Generates assets/fig1–fig5.png
│   ├── build_graph.py      # Constructs adjacency_matrix.pt
│   └── prep_2026_data.py   # Prepares subway_data_2026.csv
├── data/
│   ├── adjacency_matrix.pt       # Station graph (tracked)
│   ├── subway_data_2026.csv      # 13,759 station-day rows (not tracked)
│   ├── station_metadata.csv      # 123 stations with lat/lon (not tracked)
│   ├── manhattan_cafes.csv       # 2,457 café locations (not tracked)
│   └── weather_2026_daily.csv    # Jan–Apr 2026 NOAA daily weather (not tracked)
├── assets/                 # Generated figures (not tracked)
└── outputs/                # Model checkpoints and results (not tracked)
```

---

## Data Setup

Data files are not tracked by git. You need three inputs before running any model.

**1. Subway ridership** (`data/subway_data_2026.csv`)

Run `script/prep_2026_data.py`. Pulls from the MTA Hourly Ridership API, filters to Manhattan 7–10am for Jan–Apr 2026, and produces 13,759 station-day rows across 112 dates and 123 stations.

**2. Café density** (`data/manhattan_cafes.csv`)

Download the NYC Restaurant Inspection Results CSV from [NYC Open Data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j), place it in `script/data cleanup/`, and run `restaurant_data_cleanup.py`. Produces 2,457 Manhattan café/bakery locations.

**3. Weather** (`data/weather_2026_daily.csv`)

Download NOAA daily summaries for Central Park (station USW00094728), Jan 1–Apr 22 2026, and save as `data/weather_2026_daily.csv` with columns `date`, `tmax_c`, `prcp_mm`.

**4. Station graph** (`data/adjacency_matrix.pt`)

Already tracked. Regenerate with `python script/build_graph.py` if needed.

---

## Environment

```bash
conda create -n coffee python=3.10 pandas numpy matplotlib scikit-learn scipy
conda activate coffee
pip install torch torchvision seaborn
pip install torch_geometric
pip install "numpy<2.0" --force-reinstall
```

> `env.yaml` uses CUDA packages incompatible with Apple Silicon — use the commands above instead.

---

## Running the Models

**LSTM baselines:**
```bash
conda activate coffee
cd script/lstm\ models/
python lstm_model0.py    # ridership only
python lstm_model1.py    # ridership + café density
```
Outputs saved to `outputs/model0_outputs/` and `outputs/model1_outputs/`.

**ST-GAT (full model):**
```bash
conda activate coffee
cd script/
python train_gat.py
```
Saves `outputs/best_gat_model.pt` and `outputs/gat_results.json`. Training converges around epoch 7.

**Coffee Contribution Index:**
```bash
cd script/
python compute_cci.py
```
Saves `data/cci_scores.csv` and `outputs/cci_barplot.png`.

**Figures (for blog post):**
```bash
cd script/
python make_figures.py
```
Saves `assets/fig1.png` through `assets/fig5.png`.

---

## Reproducing Figures (Notebook)

```bash
conda activate coffee
jupyter notebook notebooks/reproduce_figures.ipynb
```

Then **Kernel → Restart & Run All**. All five blog figures will render inline. To execute headlessly and embed outputs:

```bash
jupyter nbconvert --to notebook --execute notebooks/reproduce_figures.ipynb \
  --output notebooks/reproduce_figures.ipynb
```
