import cv2
from typing import List

def extract_key_frames(video_path: str, interval: int = 10) -> List[np.ndarray]:
    """Extract frames at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            frames.append(frame)
        count += 1
    
    cap.release()
    return frames
