import sys
import os

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from bridge_mind.tcn_model import BridgeTCN


# ================= CONFIG =================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model_tcn.pth")

# ⚠️ Evaluation labels (exclude blank "_")
LABELS = [
    "HELLO",
    "YES",
    "NO",
    "THANK YOU",

    "PLEASE",
    "SORRY",
    "I LOVE YOU",
    "GOOD",
    "BAD",
    "WELCOME",

    "HELP",
    "STOP",
    "WAIT",
    "AGAIN",
    "FINISH",
    "OK",

    "WHAT",
    "WHERE",
    "WHY",
    "WHEN",
    "HOW",

    "EAT",
    "DRINK",
    "GO",
    "COME",
    "SEE",

    "NOW",
    "LATER",
    "TODAY",
    "EMERGENCY"
]
# =========================================

# shift index by 1 because model output index 0 = blank "_"
label_to_idx = {label: i + 1 for i, label in enumerate(LABELS)}
idx_to_label = {i + 1: label for i, label in enumerate(LABELS)}

device = "cpu"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = BridgeTCN(
    input_size=252,
    num_classes=len(LABELS) + 1  # +1 for blank "_"
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Trained model loaded for evaluation")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
X = []
y_true = []

for label in LABELS:
    folder = os.path.join(DATASET_DIR, label)

    if not os.path.exists(folder):
        print(f"⚠️ Skipping missing folder: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))  # (T, 252)
            X.append(data)
            y_true.append(label_to_idx[label])

if len(X) == 0:
    raise RuntimeError("No evaluation samples found.")

X = torch.from_numpy(np.array(X)).float()
y_true = np.array(y_true)

print(f"Loaded {len(X)} samples for evaluation")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
with torch.no_grad():
    logits = model(X)            # (N, T, C)
    logits = logits.mean(dim=1)  # (N, C)
    y_pred = torch.argmax(logits, dim=1).numpy()

# --------------------------------------------------
# Remove blank predictions
# --------------------------------------------------
mask = y_pred != 0
y_pred = y_pred[mask]
y_true = y_true[mask]

# --------------------------------------------------
# Metrics
# --------------------------------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(idx_to_label.keys()),
        target_names=[idx_to_label[i] for i in sorted(idx_to_label)]
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(idx_to_label.keys())
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SMART BRIDGE – Confusion Matrix (30 Signs)")
plt.tight_layout()
plt.show()
