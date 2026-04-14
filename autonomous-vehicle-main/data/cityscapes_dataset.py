import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', labels_csv=None, transform=None):
        """
        root_dir: Cityscapes root (must contain leftImg8bit/)
        split: 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.transform = transform
        self.image_paths = []
        
        if os.path.exists(self.root_dir):
            for city in os.listdir(self.root_dir):
                city_path = os.path.join(self.root_dir, city)
                if os.path.isdir(city_path):
                    for img in os.listdir(city_path):
                        if img.endswith('.png'):
                            self.image_paths.append(os.path.join(city, img))

        # Default pseudo-labels (overridable via CSV if needed, but per-image labels for Cityscapes driving are rare)
        # However, Task 1 says "overridable via a CSV"
        self.labels_dict = {}
        if labels_csv and os.path.exists(labels_csv):
            import pandas as pd
            df = pd.read_csv(labels_csv)
            # Assuming CSV has columns: image_path, steer, throttle, brake
            self.labels_dict = df.set_index('image_path').T.to_dict('list')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get labels: steer, throttle, brake
        if rel_path in self.labels_dict:
            labels = self.labels_dict[rel_path]
        else:
            labels = [0.0, 0.5, 0.0]  # default pseudo-labels

        return image, torch.tensor(labels, dtype=torch.float32)

    def __repr__(self):
        return f"CityscapesDataset(split={os.path.basename(os.path.dirname(self.root_dir))}, size={len(self)})"
