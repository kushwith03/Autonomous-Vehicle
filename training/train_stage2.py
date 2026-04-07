import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import AutoEncoder
from models.controller import ControlNet
from training.trainer import Trainer
from utils.helpers import get_device, load_config
from data.dataset import CarlaDataset
from torchvision import transforms

def train_ctrl(config_path, ae_checkpoint_path, latent_csv_path):
    cfg = load_config(config_path)
    device = get_device(cfg['device'])
    
    print(f"Loading latent data from: {latent_csv_path}")
    df = pd.read_csv(latent_csv_path)
    
    # Split features and controls
    X = df.drop(columns=["throttle", "steer", "brake"]).values
    y = df[["steer", "throttle", "brake"]].values
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Simple split for training
    split = int(len(X) * (1 - cfg['training']['stage2']['val_split']))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['stage2']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['stage2']['batch_size'])
    
    model = ControlNet(latent_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['stage2']['lr'])
    criterion = torch.nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device, cfg, "controller")
    
    print(f"Starting Controller training on {device}...")
    for epoch in range(cfg['training']['stage2']['epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Simple validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{cfg['training']['stage2']['epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    trainer.save_checkpoint(cfg['training']['stage2']['epochs'], val_loss)
    print("Stage 2 training complete.")
