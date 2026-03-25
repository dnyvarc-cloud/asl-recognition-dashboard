import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def normalize_landmarks(landmarks):
    """
    Normalizes a list of 21 landmark coordinates (x, y, z) relative to 
    the wrist (landmark 0) and scales them to a unit bounding box.
    """
    if not landmarks or len(landmarks) != 21:
        return None
        
    landmarks = np.array(landmarks)
    
    # 1. Translate so that the wrist (index 0) is at (0, 0, 0)
    wrist = landmarks[0]
    translated = landmarks - wrist
    
    # 2. Scale to a unit bounding box
    max_val = np.max(np.abs(translated))
    
    if max_val > 0:
        normalized = translated / max_val
    else:
        normalized = translated
        
    return normalized.flatten().tolist()

def extract_landmarks_from_image(image_path, detector):
    import mediapipe as mp
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        return landmarks
    return None

def process_image_dataset(image_dir, output_csv):
    """
    Processes a directory of images (structured as image_dir/Class/image.jpg) 
    and extracts landmarks using MediaPipe.
    """
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    data_rows = []
    
    # Needs the hand_landmarker.task file downloaded to models/
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    
    for label in classes:
        class_dir = os.path.join(image_dir, label)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(class_dir, img_name)
            landmarks = extract_landmarks_from_image(img_path, detector)
            
            if landmarks is not None:
                normalized = normalize_landmarks(landmarks)
                row = [label] + normalized
                data_rows.append(row)
                
    cols = ['label']
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    df = pd.DataFrame(data_rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Saved {len(df)} samples to {output_csv}")

def ingest_and_split(input_csv, output_dir, test_size=0.2, val_size=0.1):
    df = pd.read_csv(input_csv)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.concat([y_train, X_train], axis=1)
    val_df = pd.concat([y_val, X_val], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASL Dataset Preprocessing Pipeline")
    parser.add_argument('--image_dir', type=str, help='Path to directory containing images organized by class')
    parser.add_argument('--extract_csv', type=str, help='Path to save extracted landmarks CSV')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV containing landmarks to split')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Directory to save train/val/test CSVs')
    
    args = parser.parse_args()
    
    if args.image_dir and args.extract_csv:
        process_image_dataset(args.image_dir, args.extract_csv)
        
    if args.input_csv:
        ingest_and_split(args.input_csv, args.output_dir)
