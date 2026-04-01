import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import numpy as np
import time
import os
import sys
from flask import Flask, render_template, Response, jsonify, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import ASLClassifier
from data_prep import normalize_landmarks

app = Flask(__name__)

model = ASLClassifier()
model_path = 'models/asl_mlp.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    sys.exit(1)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

# Global state to synchronize accurately avoiding Ghost-Thread caching
app_state = {
    "sentence": "",
    "current_char": "",
    "confidence": 0.0,
    "latency": 0.0,
    "video_active": False,
    "prediction_buffer": [],
    "last_appended_char": "",
    "last_appended_time": time.time(),
    "last_seen_time": time.time(),
    "min_confidence_threshold": 0.70,
    "translation_paused": False
}

def gen_frames():
    global app_state
    cap = None
    
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1, running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    while True:
        if not app_state["video_active"]:
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None
                
            empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty_image, "CAMERA OFF", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', empty_image)
            
            app_state["current_char"] = "-"
            app_state["confidence"] = 0.0
            app_state["latency"] = 0.0
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start_time = time.time()
        success, image = cap.read()
        if not success:
            break
            
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        current_char = ""
        conf_score = 0.0

        if app_state["translation_paused"]:
            cv2.putText(image, "PAUSED", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            class DummyResult: hand_landmarks = []
            results = DummyResult()
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = detector.detect(mp_image)
            
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, c = image.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append([lm.x, lm.y, lm.z])
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx < x_min: x_min = cx
                    if cy < y_min: y_min = cy
                    if cx > x_max: x_max = cx
                    if cy > y_max: y_max = cy
                    
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (56, 189, 248), 3)
                
                norm_landmarks = normalize_landmarks(landmarks)
                if norm_landmarks:
                    input_tensor = torch.tensor([norm_landmarks], dtype=torch.float32)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        score, predicted = torch.max(probabilities.data, 1)
                        
                    idx = predicted.item()
                    pred_char = ' ' if idx == 26 else chr(idx + ord('A'))
                    conf_score_val = score.item()
                    
                    app_state["prediction_buffer"].append(pred_char)
                    if len(app_state["prediction_buffer"]) > 5:
                        app_state["prediction_buffer"].pop(0)
                        
                    if len(set(app_state["prediction_buffer"])) == 1 and len(app_state["prediction_buffer"]) == 5:
                        current_char = pred_char
                        conf_score = conf_score_val
                        app_state["last_seen_time"] = time.time()
                        
                        if conf_score_val >= app_state["min_confidence_threshold"]:
                            if pred_char != app_state["last_appended_char"] or (time.time() - app_state["last_appended_time"] > 1.5):
                                app_state["sentence"] += pred_char
                                app_state["last_appended_char"] = pred_char
                                app_state["last_appended_time"] = time.time()
        else:
            app_state["prediction_buffer"].clear()
            if time.time() - app_state["last_seen_time"] > 1.0:
                if len(app_state["sentence"]) > 0 and app_state["sentence"][-1] != " ":
                    app_state["sentence"] += " "
                    app_state["last_appended_char"] = " "
                    app_state["last_appended_time"] = time.time()
        
        if len(app_state["sentence"]) > 80:
            app_state["sentence"] = app_state["sentence"][-80:]
            
        app_state["current_char"] = current_char
        app_state["confidence"] = conf_score
        app_state["latency"] = (time.time() - start_time) * 1000
        
        ret, buffer = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def state():
    return jsonify(app_state)

@app.route('/clear', methods=['POST'])
def clear_text():
    app_state["sentence"] = ""
    app_state["prediction_buffer"].clear()
    app_state["last_appended_char"] = ""
    return jsonify({"status": "success"})

@app.route('/toggle_video', methods=['POST'])
def toggle_video():
    app_state["video_active"] = not app_state["video_active"]
    return jsonify({"video_active": app_state["video_active"]})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    app_state["translation_paused"] = not app_state["translation_paused"]
    return jsonify({"paused": app_state["translation_paused"]})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    data = request.json
    if data and 'threshold' in data:
        app_state["min_confidence_threshold"] = float(data['threshold'])
    return jsonify({"status": "success", "threshold": app_state["min_confidence_threshold"]})

if __name__ == '__main__':
    print("🚀 Firing up ASL Recognition Hub server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
