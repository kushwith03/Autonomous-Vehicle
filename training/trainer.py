import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, device, config, model_name):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.model_name = model_name
        
        log_path = os.path.join(config['paths']['logs_dir'], model_name)
        if log_path: os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path)
        
        self.checkpoint_dir = config['paths']['checkpoints_dir']
        if self.checkpoint_dir: os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            targets = imgs.to(self.device) if self.model_name == "autoencoder" else targets.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(imgs)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                step = epoch * len(dataloader) + i
                self.writer.add_scalar(f"Loss/train", loss.item(), step)
        
        return total_loss / len(dataloader)

    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for imgs, targets in dataloader:
                imgs = imgs.to(self.device)
                targets = imgs.to(self.device) if self.model_name == "autoencoder" else targets.to(self.device)
                
                output = self.model(imgs)
                loss = self.criterion(output, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.writer.add_scalar(f"Loss/val", avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        filename = f"{self.model_name}_best.pth" if is_best else f"{self.model_name}_latest.pth"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
        if not is_best:
            # Also save periodic ones if desired, but keep it clean
            periodic_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_ep{epoch}.pth")
            torch.save(self.model.state_dict(), periodic_path)


    def close(self):
        self.writer.close()
