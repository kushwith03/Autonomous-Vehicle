# File: E:\AutonomousVehicle\merge_and_resize.py
# Merges CARLA (kitti_converted/image_2) + Cityscapes images into a single folder
# and resizes all images to 800x600.

import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil

# --- EDIT THIS if your Cityscapes are elsewhere ---
CITYSCAPES_DIR = r"E:\AutonomousVehicle\datasets\cityscapes\leftImg8bit"

# -----------------------------------------------

CARLA_IMG_DIR = r"E:\AutonomousVehicle\datasets\kitti_converted\image_2"
OUTPUT_DIR = r"E:\AutonomousVehicle\datasets\combined\image_2"
TARGET_SIZE = (800, 600)  # width, height

os.makedirs(OUTPUT_DIR, exist_ok=True)

# start index so filenames don't collide
existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
start_idx = len(existing)

def process_and_save(src_path, dst_idx):
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        if im.size != TARGET_SIZE:
            im = im.resize(TARGET_SIZE, Image.BILINEAR)
        dst_name = f"{dst_idx:06d}.png"
        dst_path = os.path.join(OUTPUT_DIR, dst_name)
        im.save(dst_path)
        return dst_name

count = 0
idx = start_idx

# 1) copy CARLA images (already named 000000.png etc)
if os.path.isdir(CARLA_IMG_DIR):
    for name in sorted(os.listdir(CARLA_IMG_DIR)):
        if name.lower().endswith((".png",".jpg",".jpeg")):
            src = os.path.join(CARLA_IMG_DIR, name)
            process_and_save(src, idx)
            idx += 1
            count += 1
else:
    print("WARNING: CARLA image directory not found:", CARLA_IMG_DIR)

# 2) walk Cityscapes and copy images (if folder exists)
if os.path.isdir(CITYSCAPES_DIR):
    for root, _, files in os.walk(CITYSCAPES_DIR):
        for f in sorted(files):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                src = os.path.join(root, f)
                process_and_save(src, idx)
                idx += 1
                count += 1
else:
    print("NOTE: Cityscapes folder not found at", CITYSCAPES_DIR)
    print("If you have Cityscapes, edit CITYSCAPES_DIR at top of the script to the correct path and re-run.")

print("✅ Done. Total images in combined folder:", len([x for x in os.listdir(OUTPUT_DIR) if x.lower().endswith('.png')]))
print("Combined images saved to:", OUTPUT_DIR)
