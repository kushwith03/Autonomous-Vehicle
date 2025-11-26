# File: E:\AutonomousVehicle\dataloader.py
# Minimal PyTorch Dataset + DataLoader for combined images
# Usage example at bottom.

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch

class ImageFolderDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        """
        list_file: path to train.txt or val.txt (each line: image_2/000000.png)
        root_dir: parent folder containing image_2 (e.g. E:/AutonomousVehicle/datasets/combined)
        transform: torchvision transforms to apply
        """
        with open(list_file, "r") as f:
            self.items = [line.strip() for line in f if line.strip()]
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel = self.items[idx]
        img_path = os.path.join(self.root, rel.replace("/", os.sep))
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)
        else:
            # default: convert to tensor and normalize 0-1
            im = torch.from_numpy(np.array(im)).permute(2,0,1).float() / 255.0
        return im, rel  # returns (image_tensor, relative_path) — change as needed

if __name__ == "__main__":
    # quick usage test (run: python dataloader.py)
    from torchvision import transforms
    import numpy as np

    combined_root = r"E:\AutonomousVehicle\datasets\combined"
    train_list = os.path.join(combined_root, "train.txt")

    tx = transforms.Compose([
        transforms.Resize((600,800)),  # height, width (kept consistent)
        transforms.ToTensor(),
    ])

    ds = ImageFolderDataset(train_list, combined_root, transform=tx)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)

    print("Dataset size:", len(ds))
    for batch_idx, (imgs, paths) in enumerate(loader):
        print("Batch", batch_idx, "imgs.shape", imgs.shape)
        break
