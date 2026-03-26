# Class Project Template

Fill in this document at the start of the project and keep it up to date.

## 1) Project Overview

- **Learning Coffee-Driven Subway Hotspots in Manhattan**:
- **Ramya Subramanian, Sachi Patel, Hailey Gamer**:
- **Problem statement** (1-3 sentences):
- **Hypothesis** (what you expect to happen and why):

## 2) Related Work (Short)

- 3-5 bullets on prior papers/blogs/repos you build on.

## 3) Data

- **Dataset(s)**:
- **How to access** (links or scripts):
- **License/ethics**:
- **Train/val/test split**:

## 4) Baseline

- **Baseline model** LSTM Model:
- **Baseline metrics**:
- **Why this is a fair baseline**:

LSTM Model 0: Our baseline LSTM using only historical ridership failed to meaningfully separate hotspot and non-hotspot cases. Predicted probabilities clustered tightly around 0.48, indicating low confidence and weak signal. As a result, model predictions were highly sensitive to the classification threshold, flipping between predicting all non-hotspots and all hotspots. This motivated the inclusion of additional contextual features such as café density.

## 5) Proposed Method

- **What you change** (architecture, features, losses, etc.):
- **Why it should help**:
- **Ablations** (what you will remove to test impact):

## 6) Experiments

- **Metrics**:
- **Compute budget** (GPU/CPU limits, runtime):
- **Experiment plan** (list your runs and what each tests):

## 7) Reproducibility

- **How to run training**:
- **How to run evaluation**:
- **Where you log results** (files/paths):

