import os
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, device, log_dir, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, imgs) # Assuming AE for stage 1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                step = epoch * len(dataloader) + i
                self.writer.add_scalar('Loss/train', loss.item(), step)
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def save_checkpoint(self, epoch, name="checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}_{name}")
        torch.save(self.model.state_dict(), path)
        # also save as latest
        latest_path = os.path.join(self.checkpoint_dir, f"latest_{name}")
        torch.save(self.model.state_dict(), latest_path)
        print(f"Saved checkpoint: {path}")
