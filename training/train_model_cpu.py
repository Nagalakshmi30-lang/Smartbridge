import sys
import os

# --------------------------------------------------
# Add project root (smart_bridge) to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bridge_mind.tcn_model import BridgeTCN


# ================= CONFIG =================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

LABELS = [
    "_",
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

EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_model_tcn.pth")
# =========================================


label_to_idx = {label: idx for idx, label in enumerate(LABELS)}


# --------------------------------------------------
# Dataset Loader
# --------------------------------------------------
class SignDataset(Dataset):
    def __init__(self, dataset_dir):
        self.samples = []

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        for label in os.listdir(dataset_dir):
            if label not in label_to_idx:
                continue

            label_idx = label_to_idx[label]
            label_path = os.path.join(dataset_dir, label)

            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    path = os.path.join(label_path, file)
                    data = np.load(path)   # (T, 252)
                    self.samples.append((data, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No training samples found. Check dataset folder.")

        print(f"✅ Loaded {len(self.samples)} samples from dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).float(), torch.tensor(y).long()


# --------------------------------------------------
# Training Setup
# --------------------------------------------------
dataset = SignDataset(DATASET_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model = BridgeTCN(
    input_size=252,
    num_classes=len(LABELS)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = "cpu"
model.to(device)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------
print("\n🌉 SMART BRIDGE TRAINING STARTED\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)   # (B, T, 252)
        y = y.to(device)   # (B)

        optimizer.zero_grad()

        logits = model(X)            # (B, T, C)
        logits = logits.mean(dim=1)  # temporal pooling

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")


# --------------------------------------------------
# Save Trained Model
# --------------------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
print(f"💾 Model saved at: {MODEL_SAVE_PATH}")
