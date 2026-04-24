import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import AutoEncoder
from models.controller import ControlNet
from training.trainer import Trainer
from utils.helpers import get_device, load_config

def train_ctrl(config_path, latent_csv_path):
    cfg = load_config(config_path)
    device = get_device(cfg['device'])
    
    print(f"Loading latent data from: {latent_csv_path}")
    if not os.path.exists(latent_csv_path):
        print(f"[ERROR] Latent data not found: {latent_csv_path}")
        return
        
    df = pd.read_csv(latent_csv_path)
    if len(df) == 0:
        print("[ERROR] Latent data CSV is empty.")
        return
    
    # Split features and controls
    X = df.drop(columns=["throttle", "steer", "brake"]).values
    y = df[["steer", "throttle", "brake"]].values
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Simple split for training
    split = int(len(X) * (1 - cfg['training']['stage2']['val_split']))
    if split == 0 or split == len(X):
        # Handle very small datasets safely
        train_ds = TensorDataset(X, y)
        val_ds = TensorDataset(X, y)
    else:
        train_ds = TensorDataset(X[:split], y[:split])
        val_ds = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['stage2']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['stage2']['batch_size'])
    
    model = ControlNet(
        latent_dim=X.shape[1],
        hidden_dim=cfg['models']['controller']['hidden_dim'],
        output_dim=cfg['models']['controller']['output_dim']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['stage2']['lr'])
    criterion = torch.nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device, cfg, "controller")

    print(f"Starting Controller training on {device}...")
    best_loss = float('inf')

    for epoch in range(cfg['training']['stage2']['epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        print(f"Epoch [{epoch+1}/{cfg['training']['stage2']['epochs']}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(epoch + 1, val_loss, is_best=True)

    trainer.save_checkpoint(cfg['training']['stage2']['epochs'], val_loss, is_best=False)
    trainer.close()
    print("Stage 2 training complete.")
