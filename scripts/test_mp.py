import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

image_path = '/Users/danyvargas/Downloads/own_dataset/A/Image_1727698578.6699789.jpg'
image = cv2.imread(image_path)
print(f"Original shape: {image.shape}")

# Pad image
h, w = image.shape[:2]
pad_h, pad_w = h // 2, w // 2
image_padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
print(f"Padded shape: {image_padded.shape}")

image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
results = detector.detect(mp_image)
if results.hand_landmarks:
    print(f"Found {len(results.hand_landmarks)} hand(s) on padded image")
else:
    print("Still no hands found!")
