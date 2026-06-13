import sys
import os
from collections import deque

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from bridge_eye.mediapipe_tracker import BridgeEye
from bridge_sense.feature_extractor import BridgeSense
from bridge_mind.tcn_model import BridgeTCN
from bridge_voice.tts_engine import BridgeVoice


# ================= CONFIG =================
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

SEQUENCE_LENGTH = 20

CONFIDENCE_THRESHOLD = 0.70      # accuracy control
VOTING_WINDOW = 3               # majority voting
COOLDOWN_FRAMES = 10            # debounce

MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model_tcn.pth")
# =========================================


device = "cpu"

# --------------------------------------------------
# Init components
# --------------------------------------------------
eye = BridgeEye()
sense = BridgeSense()
voice = BridgeVoice()

model = BridgeTCN(
    input_size=252,
    num_classes=len(LABELS)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Trained model loaded")


# --------------------------------------------------
# Runtime buffers
# --------------------------------------------------
frame_buffer = []
prediction_queue = deque(maxlen=VOTING_WINDOW)

last_spoken = None
cooldown = 0

cap = cv2.VideoCapture(0)
print("🌉 SMART BRIDGE running... Press ESC to exit")


# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = eye.extract_landmarks(frame)
    features = sense.extract_features(landmarks)
    frame_buffer.append(features)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        seq = torch.from_numpy(np.array(frame_buffer)).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(seq)[0]        # (T, C)
            logits = logits.mean(dim=0)   # (C)
            probs = F.softmax(logits, dim=0)

        conf, pred_idx = torch.max(probs, dim=0)
        pred_label = LABELS[pred_idx.item()]

        # ---- Confidence filter ----
        if conf.item() > CONFIDENCE_THRESHOLD and pred_label != "_":
            prediction_queue.append(pred_label)

        frame_buffer = []

    # ---- Majority voting ----
    if len(prediction_queue) == VOTING_WINDOW:
        final_pred = max(set(prediction_queue), key=prediction_queue.count)

        if final_pred != last_spoken and cooldown == 0:
            print(f"SMART BRIDGE: {final_pred}")
            voice.speak(final_pred)

            last_spoken = final_pred
            cooldown = COOLDOWN_FRAMES
            prediction_queue.clear()

    if cooldown > 0:
        cooldown -= 1

    cv2.imshow("SMART BRIDGE", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
