import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CarlaDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None, labels_csv=None):
        if not os.path.exists(list_file):
            print(f"Warning: List file not found: {list_file}")
            self.items = []
        else:
            with open(list_file, "r") as f:
                self.items = [line.strip() for line in f if line.strip()]
        
        self.root = root_dir
        self.transform = transform
        self.labels = None
        if labels_csv and os.path.exists(labels_csv):
            self.labels = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path = self.items[idx]
        img_path = os.path.join(self.root, rel_path.replace("/", os.sep))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            row = self.labels.iloc[idx]
            controls = torch.tensor([row['steer'], row['throttle'], row['brake']], dtype=torch.float32)
            return image, controls
            
        return image, rel_path
