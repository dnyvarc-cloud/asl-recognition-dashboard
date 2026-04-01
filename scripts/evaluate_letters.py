import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.train import ASLClassifier, ASLDataset

def evaluate_best_letter():
    model_path = 'models/asl_mlp.pth'
    test_csv = 'data/processed/test.csv'
    
    # Load dataset & model
    test_dataset = ASLDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ASLClassifier(input_size=63, hidden_sizes=[256, 128], num_classes=27)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    class_correct = {i: 0 for i in range(27)}
    class_total = {i: 0 for i in range(27)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            for p, l in zip(predicted, labels):
                if p == l:
                    class_correct[l.item()] += 1
                class_total[l.item()] += 1
                
    results = []
    for class_id in range(27):
        if class_total[class_id] > 0:
            char_key = 'space' if class_id == 26 else chr(class_id + ord('A'))
            acc = class_correct[class_id] / class_total[class_id] * 100
            results.append((char_key, acc, class_total[class_id]))
            
    # Sort by accuracy then by total counts
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("Test Accuracy per Category:")
    for char, acc, total in results:
        print(f"[{char}] -> {acc:.1f}% ({total} samples)")
        
if __name__ == '__main__':
    evaluate_best_letter()
