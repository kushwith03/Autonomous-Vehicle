import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import CarlaDataset
from models.autoencoder import AutoEncoder
from training.trainer import Trainer
from utils.helpers import get_device, load_config

def train_ae(config_path):
    cfg = load_config(config_path)
    device = get_device(cfg['device'])
    
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg['models']['autoencoder']['img_size'])),
        transforms.ToTensor(),
    ])
    
    dataset = CarlaDataset(cfg['paths']['train_list'], cfg['paths']['data_root'], transform)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check your paths in config.")
        return

    loader = DataLoader(dataset, batch_size=cfg['training']['stage1']['batch_size'], shuffle=True)
    
    model = AutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['stage1']['lr'])
    criterion = torch.nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device, cfg, "autoencoder")
    
    print(f"Starting AutoEncoder training on {device}...")
    for epoch in range(cfg['training']['stage1']['epochs']):
        loss = trainer.train_epoch(loader, epoch)
        print(f"Epoch [{epoch+1}/{cfg['training']['stage1']['epochs']}] Avg Loss: {loss:.4f}")
        if (epoch + 1) % cfg['training']['stage1']['save_every'] == 0:
            trainer.save_checkpoint(epoch + 1, loss)
    
    trainer.close()
    print("Stage 1 training complete.")
