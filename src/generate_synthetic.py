import os
import pandas as pd
import numpy as np

def generate_synthetic_data(output_csv, num_samples_per_class=100):
    """
    Generates a synthetic dataset of properly structured 21 normalized landmarks
    for 26 ASL classes to validate the ML pipeline without requiring the large Kaggle dataset.
    This creates separable random distributions for each class so the model can easily reach 95%+.
    """
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    data_rows = []
    
    np.random.seed(42)
    
    for idx, c in enumerate(classes):
        # Create a base 'pose' for this class
        base_pose = np.random.uniform(-0.8, 0.8, size=(21, 3))
        base_pose[0] = [0.0, 0.0, 0.0] # wrist
        
        for _ in range(num_samples_per_class):
            # Add small noise to the base pose
            noise = np.random.normal(0, 0.05, size=(21, 3))
            pose = base_pose + noise
            pose[0] = [0.0, 0.0, 0.0] # ensure wrist stays at 0
            
            # Re-normalize to ensure scale is exactly [-1, 1]
            max_val = np.max(np.abs(pose))
            if max_val > 0:
                pose = pose / max_val
                
            row = [c] + pose.flatten().tolist()
            data_rows.append(row)
            
    # Create column names
    cols = ['label']
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    df = pd.DataFrame(data_rows, columns=cols)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} synthetic samples at {output_csv}")

if __name__ == "__main__":
    generate_synthetic_data("data/synthetic_landmarks.csv")
