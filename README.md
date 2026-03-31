# Morning Brews & Morning Rush: Coffee-Driven Subway Hotspots in Manhattan

STAT W3106 – Spring 2026 | Ramya Subramanian, Sachi Patel, Hailey Gamer

## Overview

We model 7–10am Manhattan subway "hotspots" (top 25% of morning ridership by station-day) as a spatio-temporal prediction problem, benchmarking LSTM baselines against a full Spatio-Temporal Graph Attention Network (GAT) + GRU.

## Data Setup

Data is stored locally and not tracked by git. Generate both files before running models.

**Subway ridership** (`data/subway_data.csv`): Run `script/data cleanup/subway_data_cleanup.py`. Pulls ~2M rows from the MTA Hourly Ridership API, filters to Manhattan 7–10am, and produces ~9,400 station-day rows across 95 dates and 123 stations.

**Café density** (`data/manhattan_cafes.csv`): Download the NYC Restaurant Inspection Results CSV (dated 20260325) from [NYC Open Data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j), place it in `script/data cleanup/`, then run `restaurant_data_cleanup.py`. Produces 2,457 Manhattan café/bakery locations.

## Environment
```bash
conda create -n neds python=3.10 pandas numpy matplotlib scikit-learn scipy
conda activate neds
pip install torch torchvision seaborn
pip install "numpy<2.0" --force-reinstall
```

Note: `env.yaml` uses CUDA packages incompatible with Apple Silicon — use the above instead.

## Running the Models
```bash
conda activate neds
cd script/lstm\ models/
python lstm_model0.py
python lstm_model1.py
```

Outputs saved to `outputs/model0_outputs/` and `outputs/model1_outputs/`.

## Baseline Results

| Model | Features | F1 | AUC |
|-------|----------|----|-----|
| Model 0 | Ridership only (3-day sequence) | 0.48 | 0.71 |
| Model 1 | Ridership + café density (7-day sequence) | 0.59 | 0.79 |

Adding café density as a static station-level feature improved every metric, confirming that third-place density adds predictive signal beyond ridership history alone. Both models treat stations independently but the GAT addresses this by learning spatial spillover across the station graph.
