import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SUBWAY_FILE = "../../data/subway_data.csv"
OUTPUT_DIR = "../../outputs/model0_outputs"

SEQUENCE_LENGTH = 3
TRAIN_FRAC = 0.80
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32
NUM_LAYERS = 1
RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)

def load_data():
    df = pd.read_csv(SUBWAY_FILE, low_memory=False)

    required_cols = [
        "date",
        "station_complex_id",
        "station_complex",
        "latitude",
        "longitude",
        "morning_ridership",
        "hotspot",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in subway_data.csv: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["station_complex_id"] = df["station_complex_id"].astype(str)
    df["hotspot"] = df["hotspot"].astype(int)

    df = df.dropna(subset=required_cols).copy()
    df = df.sort_values(["station_complex_id", "date"]).reset_index(drop=True)

    return df

def get_cutoff_date(df, train_frac=0.8):
    unique_dates = sorted(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * train_frac)
    cutoff_idx = min(max(cutoff_idx, 1), len(unique_dates) - 1)
    return pd.to_datetime(unique_dates[cutoff_idx])

def scale_ridership(df, cutoff_date):
    df = df.copy()
    train_mask = df["date"] < cutoff_date

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, ["morning_ridership"]])

    df["morning_ridership_scaled"] = scaler.transform(df[["morning_ridership"]])
    return df

def build_sequences(df, sequence_length=7):
    X, y, meta = [], [], []

    for station_id, group in df.groupby("station_complex_id"):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) <= sequence_length:
            continue

        feature_array = group[["morning_ridership_scaled"]].to_numpy(dtype=np.float32)
        target_array = group["hotspot"].to_numpy(dtype=np.float32)

        for i in range(sequence_length, len(group)):
            seq = feature_array[i-sequence_length:i]
            target = target_array[i]

            X.append(seq)
            y.append(target)
            meta.append({
                "station_complex_id": station_id,
                "station_complex": group.loc[i, "station_complex"],
                "date": group.loc[i, "date"],
            })

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    meta = pd.DataFrame(meta)

    return X, y, meta

def split_by_time(X, y, meta, cutoff_date):
    mask_train = pd.to_datetime(meta["date"]) < cutoff_date
    mask_test = ~mask_train

    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    meta_train = meta.loc[mask_train].reset_index(drop=True)
    meta_test = meta.loc[mask_test].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_train, meta_test

def train_model(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    if positive_count > 0:
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f}")

    model.eval()
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.475).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().astype(int).tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    metrics = {
        "accuracy": accuracy_score(all_true, all_preds),
        "precision": precision_score(all_true, all_preds, zero_division=0),
        "recall": recall_score(all_true, all_preds, zero_division=0),
        "f1": f1_score(all_true, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_true, all_probs) if len(np.unique(all_true)) > 1 else np.nan,
    }

    return model, train_losses, all_true, all_preds, all_probs, metrics

def plot_example_sequence(X, y):
    seq = X[0, :, 0]
    target = int(y[0])

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(seq) + 1), seq, marker="o")
    plt.xlabel("Day in input sequence")
    plt.ylabel("Scaled ridership")
    plt.title(f"Model 0 Example Sequence | Next-day hotspot = {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_example_sequence.png"), dpi=200)
    plt.close()

def plot_training_loss(train_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Model 0 Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_training_loss.png"), dpi=200)
    plt.close()

def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Model 0 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_confusion_matrix.png"), dpi=200)
    plt.close()

def plot_roc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model 0 ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_roc_curve.png"), dpi=200)
    plt.close()

def plot_probability_histogram(y_prob):
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob, bins=20)
    plt.xlabel("Predicted hotspot probability")
    plt.ylabel("Count")
    plt.title("Model 0 Predicted Probability Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_probability_histogram.png"), dpi=200)
    plt.close()

def plot_actual_vs_predicted(meta_test, y_true, y_pred):
    df_plot = meta_test.copy()
    df_plot["actual"] = y_true
    df_plot["predicted"] = y_pred
    df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")

    df_plot = df_plot.dropna(subset=["date"]).copy()

    summary = df_plot.groupby("date")[["actual", "predicted"]].mean().reset_index()

    print("\nActual vs Predicted summary:")
    print(summary)

    if len(summary) == 0:
        print("No rows available for actual vs predicted plot.")
        return

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, summary["actual"], width=width, label="Actual hotspot rate")
    plt.bar(x + width/2, summary["predicted"], width=width, label="Predicted hotspot rate")

    plt.xlabel("Date")
    plt.ylabel("Share of stations labeled hotspot")
    plt.title("Model 0 Actual vs Predicted Hotspot Rate")
    plt.xticks(x, summary["date"].dt.strftime("%Y-%m-%d"), rotation=45)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_actual_vs_predicted.png"), dpi=200)
    plt.close()

def plot_top_hotspot_stations(df):
    hotspot_rows = df[df["hotspot"] == 1].copy()

    summary = (
        hotspot_rows.groupby(["station_complex_id", "station_complex"], as_index=False)
        .agg(hotspot_days=("hotspot", "count"))
        .sort_values("hotspot_days", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(summary["station_complex"], summary["hotspot_days"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Number of hotspot days")
    plt.title("Top 10 Most Frequent Hotspot Stations")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model0_top_hotspot_stations.png"), dpi=200)
    plt.close()

    summary.to_csv(os.path.join(OUTPUT_DIR, "model0_top_hotspot_stations.csv"), index=False)

def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    cutoff_date = get_cutoff_date(df, TRAIN_FRAC)
    df = scale_ridership(df, cutoff_date)

    X, y, meta = build_sequences(df, SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_by_time(X, y, meta, cutoff_date)

    print("Cutoff date:", cutoff_date.date())
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    model, train_losses, y_true, y_pred, y_prob, metrics = train_model(X_train, y_train, X_test, y_test)

    print("\nModel 0 Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model0_metrics.csv"), index=False)

    pred_df = meta_test.copy()
    pred_df["actual"] = y_true
    pred_df["predicted"] = y_pred
    pred_df["predicted_probability"] = y_prob
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "model0_test_predictions.csv"), index=False)

    plot_example_sequence(X_train, y_train)
    plot_training_loss(train_losses)
    plot_conf_matrix(y_true, y_pred)
    plot_roc(y_true, y_prob)
    plot_probability_histogram(y_prob)
    plot_actual_vs_predicted(meta_test, y_true, y_pred)
    plot_top_hotspot_stations(df)

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()