import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.carla_dataset import CarlaDataset
from data.cityscapes_dataset import CityscapesDataset
from data.combined_dataset import CombinedDataset, get_dataset_stats
from models.autoencoder import AutoEncoder
from utils.trainer import Trainer

def train_stage1(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Stage 1 (AutoEncoder) on: {device}")

    transform = transforms.Compose([
        transforms.Resize(tuple(cfg['models']['autoencoder']['input_dim'][1:])),
        transforms.ToTensor(),
    ])

    if cfg['dataset']['use_combined']:
        carla_ds = CarlaDataset(cfg['paths']['carla_root'], transform=transform)
        cityscapes_ds = CityscapesDataset(cfg['paths']['cityscapes_root'], transform=transform)
        dataset = CombinedDataset(carla_ds, cityscapes_ds)
        stats = get_dataset_stats(dataset)
        print(f"📊 Dataset Stats: {stats}")
    else:
        dataset = CarlaDataset(cfg['paths']['carla_root'], transform=transform)
        print(f"📊 Dataset size: {len(dataset)} (CARLA only)")

    dataloader = DataLoader(dataset, batch_size=cfg['training']['stage1']['batch_size'], shuffle=True, num_workers=2)

    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['stage1']['lr'])
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        log_dir=cfg['paths']['logs'],
        checkpoint_dir=cfg['paths']['checkpoints']
    )

    epochs = cfg['training']['stage1']['epochs']
    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch, is_ae=True)
        print(f"✅ Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1, name="autoencoder.pth")

    print("🏁 Training complete.")

if __name__ == "__main__":
    train_stage1("configs/default_config.yaml")
