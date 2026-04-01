import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class ASLDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        def label_to_id(c):
            if str(c).lower() == 'space': return 26
            return ord(str(c)) - ord('A')
        self.labels = [label_to_id(c) for c in df['label']]
        self.features = df.drop('label', axis=1).values
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_sizes=[256, 128], num_classes=27):
        super(ASLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def train_model(data_dir, model_save_path, epochs=20, lr=0.001):
    train_dataset = ASLDataset(os.path.join(data_dir, 'train.csv'))
    val_dataset = ASLDataset(os.path.join(data_dir, 'val.csv'))
    test_dataset = ASLDataset(os.path.join(data_dir, 'test.csv'))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ASLClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_dataset):.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            
    # Final Test
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = correct / total
    print(f"Final Test Accuracy: {test_acc:.4f}")
if __name__ == '__main__':
    train_model('data/processed', 'models/asl_mlp.pth', epochs=150)
