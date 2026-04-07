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
        self.writer = SummaryWriter(os.path.join(config['paths']['logs_dir'], model_name))
        self.checkpoint_dir = config['paths']['checkpoints_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            elif isinstance(targets, list):
                # AutoEncoder case where targets are paths, we use imgs as targets
                targets = imgs
            else:
                targets = imgs

            self.optimizer.zero_grad()
            output = self.model(imgs)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                step = epoch * len(dataloader) + i
                self.writer.add_scalar(f"Loss/train_{self.model_name}", loss.item(), step)
        
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch, loss):
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}_loss{loss:.4f}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Checkpoint saved to {path}")
