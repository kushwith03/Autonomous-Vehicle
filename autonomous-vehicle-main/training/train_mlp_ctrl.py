import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.controller import ControlNet
from utils.trainer import Trainer

def train_mlp_ctrl(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training MLP Controller on: {device}")

    # Load latent CSV
    latent_csv = "datasets/val_latents.csv"
    if not os.path.exists(latent_csv):
        print(f"Error: Latent CSV not found at {latent_csv}. Run extract_features first.")
        return
        
    df = pd.read_csv(latent_csv)
    X = np.array([list(map(float, x.split())) for x in df['latent']])
    y = df[['steer', 'throttle', 'brake']].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ControlNet(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    # Use a dummy Trainer that works for TensorDataset
    # Wait, Trainer works fine with any dataloader that yields (imgs, labels) 
    # for TensorDataset it yields (X, y) which is the same.
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        log_dir=cfg['paths']['logs'],
        checkpoint_dir=cfg['paths']['checkpoints']
    )

    epochs = 20
    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch, is_ae=False)
        print(f"✅ Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1, name="mlp_ctrl.pth")

    print("🏁 Training complete.")

if __name__ == "__main__":
    train_mlp_ctrl("configs/default_config.yaml")
