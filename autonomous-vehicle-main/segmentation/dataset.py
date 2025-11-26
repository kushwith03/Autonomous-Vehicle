import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CityscapesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=(256, 512), augment=True):
        """
        Loads images and label masks.
        size = (height, width)
        augment = whether to apply random flips / brightness / contrast.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.png', '.jpg'))])
        self.size = size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize(size, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ----- Data augmentation -----
        if self.augment:
            # random horizontal flip
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            # random brightness and contrast
            if random.random() > 0.7:
                img = transforms.functional.adjust_brightness(img, 1.2)
            if random.random() > 0.7:
                img = transforms.functional.adjust_contrast(img, 1.2)
        # -----------------------------

        img_t = self.base_transform(img)
        mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        return img_t, mask_np, fname
