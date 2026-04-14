import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, root_dir, labels_csv=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = None
        if labels_csv and os.path.exists(labels_csv):
            self.labels_df = pd.read_csv(labels_csv)
            # assume CSV has columns: image_path, steer, throttle, brake
            self.image_paths = self.labels_df['image_path'].tolist()
        else:
            # fallback: load all images in root_dir if no CSV
            self.image_paths = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels_df is not None:
            labels = self.labels_df.iloc[idx][['steer', 'throttle', 'brake']].values.astype('float32')
            return image, torch.tensor(labels)
        else:
            # default labels if none provided (useful for AE training)
            return image, torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)

    def __repr__(self):
        return f"CarlaDataset(size={len(self)})"
