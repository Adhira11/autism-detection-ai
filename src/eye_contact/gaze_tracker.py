import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.L_EYE = [33, 133]
        self.R_EYE = [362, 263]

    def detect_gaze(self, frame: np.ndarray) -> Optional[float]:
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        left_eye = landmarks.landmark[self.L_EYE[0]]
        right_eye = landmarks.landmark[self.R_EYE[0]]
        
        return (left_eye.z + right_eye.z) / 2  # Simplified gaze score
