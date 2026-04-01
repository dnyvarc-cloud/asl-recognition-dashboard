import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import numpy as np
import time
import os
import sys

# Add the src dir to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import ASLClassifier
from data_prep import normalize_landmarks

# Load Model
model = ASLClassifier()
model_path = 'models/asl_mlp.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    sys.exit(1)
    
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prediction_buffer = []
    display_text = ""
    
    # Initialize MediaPipe Tasks API
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Drawing utils from original library if available, else manual drawing
    try:
        from mediapipe.python.solutions import drawing_utils
        from mediapipe.python.solutions import hands as mp_hands_old
        from mediapipe.framework.formats import landmark_pb2
        has_drawing_utils = True
    except (ImportError, AttributeError):
        has_drawing_utils = False

    sentence = ""
    last_appended_char = ""
    last_appended_time = time.time()
    last_seen_time = time.time()

    while True:
        start_time = time.time()
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
            
        image = cv2.flip(image, 1) # selfies
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks
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
                    
                # Optionally draw skeleton
                if has_drawing_utils:
                    # Convert HandLandmarker output to protobuf format if using drawing_utils
                    landmark_list_pb2 = landmark_pb2.NormalizedLandmarkList()
                    landmark_list_pb2.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                    ])
                    drawing_utils.draw_landmarks(
                        image,
                        landmark_list_pb2,
                        mp_hands_old.HAND_CONNECTIONS)
                else:
                    # Fallback minimal visualization: draw points
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
                    
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)
                
                norm_landmarks = normalize_landmarks(landmarks)
                if norm_landmarks:
                    input_tensor = torch.tensor([norm_landmarks], dtype=torch.float32)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        score, predicted = torch.max(probabilities.data, 1)
                        
                    idx = predicted.item()
                    pred_char = ' ' if idx == 26 else chr(idx + ord('A'))
                    conf_score = score.item()
                    
                    # Temporal Smoothing
                    prediction_buffer.append(pred_char)
                    if len(prediction_buffer) > 5:
                        prediction_buffer.pop(0)
                        
                    if len(set(prediction_buffer)) == 1 and len(prediction_buffer) == 5:
                        display_text = f"{pred_char} ({conf_score*100:.1f}%)"
                        last_seen_time = time.time()
                        
                        if pred_char != last_appended_char or (time.time() - last_appended_time > 1.5):
                            sentence += pred_char
                            last_appended_char = pred_char
                            last_appended_time = time.time()
                        
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, display_text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # No hands detected, clear smoothing buffer
            prediction_buffer.clear()
            display_text = ""
            
            if time.time() - last_seen_time > 1.0:
                if len(sentence) > 0 and sentence[-1] != " ":
                    sentence += " "
                    last_appended_char = " "
                    last_appended_time = time.time()
            
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        cv2.putText(image, f"Latency: {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
        # Render the sentence at the bottom
        if len(sentence) > 60:
            sentence = sentence[-60:]
            
        line1 = sentence[:30]
        line2 = sentence[30:]
        
        # Transparent background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 400), (640, 480), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        cv2.putText(image, line1, (10, 435), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if line2:
            cv2.putText(image, line2, (10, 470), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
        cv2.imshow('ASL Real-Time Translator', image)
        
        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
