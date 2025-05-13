import mediapipe as mp
import numpy as np
from scipy.fft import fft

class MotionAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.history = []
        
    def detect_repetition(self, frame: np.ndarray) -> bool:
        results = self.pose.process(frame)
        if not results.pose_landmarks:
            return False
            
        wrist = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        self.history.append((wrist.x, wrist.y))
        
        if len(self.history) < 30:  # config.yaml value
            return False
            
        x_coords = [x for x, _ in self.history[-30:]]
        y_coords = [y for _, y in self.history[-30:]]
        
        x_freq = np.abs(fft(x_coords)[1:10])
        y_freq = np.abs(fft(y_coords)[1:10])
        
        return np.max(x_freq) > 10 or np.max(y_freq) > 10  # Threshold from config
