import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import CarlaDataset
from models.cnn_controller import CNNController
from training.trainer import Trainer
from utils.helpers import load_config, get_device

def train_cnn_bc(config_path):
    cfg = load_config(config_path)
    device = get_device(cfg.get('device', 'auto'))
    print(f"🚀 Training CNN Behavioral Cloning on: {device}")

    # For CNN BC, we typically use the same resolution as AE (128x128)
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg['models']['autoencoder']['img_size'])),
        transforms.ToTensor(),
    ])

    # Use CarlaDataset with paths from config
    dataset = CarlaDataset(
        list_file=cfg['paths']['train_list'],
        root_dir=cfg['paths']['data_root'],
        transform=transform,
        labels_csv=cfg['paths'].get('carla_labels_csv')
    )
    
    print(f"📊 Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['training']['cnn_bc']['batch_size'], 
        shuffle=True, 
        num_workers=0 # Stable for all OS
    )

    model = CNNController(
        output_dim=cfg['models']['cnn_controller']['output_dim'],
        dropout=cfg['models']['cnn_controller']['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['cnn_bc']['lr'])
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        model_name="cnn_bc"
    )

    epochs = cfg['training']['cnn_bc']['epochs']
    save_every = cfg['training']['cnn_bc']['save_every']
    
    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"✅ Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, avg_loss)

    print("🏁 Training complete.")


if __name__ == "__main__":
    train_cnn_bc("configs/default_config.yaml")
