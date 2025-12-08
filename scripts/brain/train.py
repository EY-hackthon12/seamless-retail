import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from model import RetailSalesPredictor
import argparse

class RetailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "retail_data.csv")
    print(f"--> Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    # Features: DayOfWeek, IsWeekend, IsHoliday, Promo, Rainfall, Footfall, Inventory
    feature_cols = ["DayOfWeek", "IsWeekend", "IsHoliday", "Promo", "Rainfall", "Footfall", "Inventory"]
    target_col = "Sales"
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # 2. Normalize (StandardScaler)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8 # Avoid div by zero
    X_scaled = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_scaled = (y - y_mean) / y_std
    
    # Save scalers for inference
    scaler_stats = {
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "features": feature_cols
    }
    with open(os.path.join(os.path.dirname(__file__), "scaler.json"), "w") as f:
        json.dump(scaler_stats, f)
        
    dataset = RetailDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Init Model
    model = RetailSalesPredictor(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("--> Starting training...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 4. Save Model
    save_path = os.path.join(os.path.dirname(__file__), "sales_brain.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--> Model saved to {save_path}")

if __name__ == "__main__":
    train()
