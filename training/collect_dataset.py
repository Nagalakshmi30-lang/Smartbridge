import sys
import os

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from datetime import datetime

from bridge_eye.mediapipe_tracker import BridgeEye
from bridge_sense.feature_extractor import BridgeSense


# ================= CONFIG =================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SEQUENCE_LENGTH = 30

SIGNS = [
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


eye = BridgeEye()
sense = BridgeSense()

cap = cv2.VideoCapture(0)

# --------------------------------------------------
# Ensure dataset folders exist
# --------------------------------------------------
os.makedirs(DATASET_DIR, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(DATASET_DIR, sign), exist_ok=True)

print("\n🌉 SMART BRIDGE – FLEXIBLE DATASET COLLECTION")
print("Instructions:")
print("  c → collect data for this sign")
print("  s → skip this sign")
print("  q → quit program\n")

# --------------------------------------------------
# Main interactive loop
# --------------------------------------------------
for sign in SIGNS:
    print(f"\nSign: {sign}")
    choice = input("Enter choice [c/s/q]: ").strip().lower()

    if choice == "q":
        print("Exiting dataset collection.")
        break

    if choice == "s":
        print(f"Skipped sign: {sign}")
        continue

    if choice != "c":
        print("Invalid input, skipping sign.")
        continue

    try:
        samples = int(input(f"How many samples to collect for {sign}? "))
    except ValueError:
        print("Invalid number. Skipping.")
        continue

    count = 0
    while count < samples:
        frames = []
        print(f"Recording {sign} | Sample {count+1}/{samples}")

        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            landmarks = eye.extract_landmarks(frame)
            features = sense.extract_features(landmarks)
            frames.append(features)

            cv2.putText(
                frame,
                f"{sign} | Frames {len(frames)}/{SEQUENCE_LENGTH}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("SMART BRIDGE - Dataset Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Sample skipped")
                frames = []
                break

        if len(frames) == SEQUENCE_LENGTH:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_{timestamp}_{count}.npy"
            filepath = os.path.join(DATASET_DIR, sign, filename)

            np.save(filepath, np.array(frames))
            print(f"Saved → {filepath}")
            count += 1

    print(f"Finished collecting for sign: {sign}")

print("\n✅ Dataset collection finished.")
cap.release()
cv2.destroyAllWindows()
