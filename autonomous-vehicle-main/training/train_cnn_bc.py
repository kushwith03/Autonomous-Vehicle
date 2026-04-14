import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.carla_dataset import CarlaDataset
from models.cnn_controller import CNNController
from utils.trainer import Trainer

def train_cnn_bc(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training CNN Behavioral Cloning on: {device}")

    # For CNN BC, we typically use the same resolution as AE (128x128)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Task 2: Use CarlaDataset with labels_csv loaded from config
    dataset = CarlaDataset(
        root_dir=cfg['paths']['carla_root'],
        labels_csv=cfg['paths']['carla_labels_csv'],
        transform=transform
    )
    
    print(f"📊 Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['training']['cnn_bc']['batch_size'], 
        shuffle=True, 
        num_workers=2
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
        log_dir=cfg['paths']['logs'],
        checkpoint_dir=cfg['paths']['checkpoints']
    )

    epochs = cfg['training']['cnn_bc']['epochs']
    save_every = cfg['training']['cnn_bc']['save_every']
    
    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch, is_ae=False)
        print(f"✅ Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, name="cnn_bc.pth")

    print("🏁 Training complete.")

if __name__ == "__main__":
    train_cnn_bc("configs/default_config.yaml")
