import cv2
import mediapipe as mp
import numpy as np

class BridgeEye:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(image)

        keypoints = []

        def collect(points):
            if points:
                for p in points.landmark:
                    keypoints.extend([p.x, p.y, p.z])
            else:
                keypoints.extend([0.0] * 63)

        collect(result.left_hand_landmarks)
        collect(result.right_hand_landmarks)

        return np.array(keypoints, dtype=np.float32)

    def draw(self, frame, result):
        if result.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, result.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)

        if result.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, result.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)
