import numpy as np

class BridgeSense:
    def __init__(self):
        self.prev = None

    def normalize(self, landmarks):
        landmarks = landmarks.reshape(-1, 3)

        wrist = landmarks[0]
        landmarks = landmarks - wrist

        scale = np.linalg.norm(landmarks[9]) + 1e-6
        landmarks = landmarks / scale

        return landmarks.flatten()

    def velocity(self, current):
        if self.prev is None:
            vel = np.zeros_like(current)
        else:
            vel = current - self.prev
        self.prev = current
        return vel

    def extract_features(self, landmarks):
        norm = self.normalize(landmarks)
        vel = self.velocity(norm)
        return np.concatenate([norm, vel])
