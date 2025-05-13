import pytest
import cv2
import numpy as np
from src.eye_contact.gaze_tracker import GazeTracker

@pytest.fixture
def tracker():
    return GazeTracker()

def test_gaze_detection(tracker):
    # Test with blank frame (should return None)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert tracker.detect_gaze(blank_frame) is None
    
    # Test with mock face (simplified)
    mock_frame = cv2.imread('tests/test_face.jpg')  # Add sample face image
    assert 0 <= tracker.detect_gaze(mock_frame) <= 1
