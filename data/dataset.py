import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None, labels_csv=None, return_path=False):
        if not os.path.exists(list_file):
            print(f"Warning: List file not found: {list_file}")
            self.items = []
        else:
            with open(list_file, "r") as f:
                self.items = [line.strip() for line in f if line.strip()]
        
        self.root = root_dir
        self.transform = transform
        self.label_map = {}
        self.return_path = return_path
        
        if labels_csv and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            # Use basename of 'image' column to map to controls
            df['basename'] = df['image'].apply(lambda x: os.path.basename(str(x)))
            self.label_map = df.drop_duplicates('basename').set_index('basename')[['steer', 'throttle', 'brake']].to_dict('index')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path = self.items[idx]
        img_path = os.path.join(self.root, rel_path.replace("/", os.sep))
        filename = os.path.basename(rel_path)
        
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)
        
        if filename in self.label_map:
            vals = self.label_map[filename]
            controls = torch.tensor([vals['steer'], vals['throttle'], vals['brake']], dtype=torch.float32)
        else:
            # Consistent output for DataLoader even if labels are missing (e.g. for Stage 1)
            controls = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            
        if self.return_path:
            return image, controls, rel_path
        return image, controls
