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

    # Split for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['stage1']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['stage1']['batch_size'])
    
    model = AutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['stage1']['lr'])
    criterion = torch.nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device, cfg, "autoencoder")
    
    print(f"Starting AutoEncoder training on {device}...")
    best_loss = float('inf')
    
    for epoch in range(cfg['training']['stage1']['epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)
        
        print(f"Epoch [{epoch+1}/{cfg['training']['stage1']['epochs']}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(epoch + 1, val_loss, is_best=True)
            
        if (epoch + 1) % cfg['training']['stage1']['save_every'] == 0:
            trainer.save_checkpoint(epoch + 1, val_loss, is_best=False)

    
    trainer.close()
    print("Stage 1 training complete.")
