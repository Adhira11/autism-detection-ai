from flask import Flask, request, jsonify
from src.eye_contact.gaze_tracker import GazeTracker
import cv2
import tempfile
import yaml

app = Flask(__name__)
tracker = GazeTracker()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
        
    video = request.files['video']
    temp_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    video.save(temp_path)
    
    cap = cv2.VideoCapture(temp_path)
    scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        score = tracker.detect_gaze(frame)
        if score is not None:
            scores.append(score)
    
    cap.release()
    avg_score = np.mean(scores) if scores else 0
    
    return jsonify({
        "gaze_score": avg_score,
        "diagnosis": "Low" if avg_score < config['gaze']['threshold'] else "Normal"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
