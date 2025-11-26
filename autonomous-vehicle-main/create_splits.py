# File: E:\AutonomousVehicle\create_splits.py
# Creates train.txt and val.txt listing relative image paths (for combined dataset)
# Usage: python E:\AutonomousVehicle\create_splits.py

import os
import random

RANDOM_SEED = 42
COMBINED_DIR = r"E:\AutonomousVehicle\datasets\combined\image_2"
OUTPUT_DIR = r"E:\AutonomousVehicle\datasets\combined"
TRAIN_RATIO = 0.8   # 80% train, 20% val

random.seed(RANDOM_SEED)

# collect images
imgs = sorted([f for f in os.listdir(COMBINED_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
total = len(imgs)
if total == 0:
    raise SystemExit("No images found in: " + COMBINED_DIR)

# shuffle deterministically
indices = list(range(total))
random.shuffle(indices)

# split
split_idx = int(total * TRAIN_RATIO)
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

# write files (relative paths from combined folder)
train_file = os.path.join(OUTPUT_DIR, "train.txt")
val_file = os.path.join(OUTPUT_DIR, "val.txt")

with open(train_file, "w") as f:
    for i in train_idx:
        f.write(os.path.join("image_2", imgs[i]).replace("\\","/") + "\n")

with open(val_file, "w") as f:
    for i in val_idx:
        f.write(os.path.join("image_2", imgs[i]).replace("\\","/") + "\n")

print(f"Total images: {total}")
print(f"Train: {len(train_idx)}")
print(f"Val:   {len(val_idx)}")
print("Wrote:", train_file)
print("Wrote:", val_file)
