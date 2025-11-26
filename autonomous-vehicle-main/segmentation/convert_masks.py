# convert_masks.py
import os
import numpy as np
from PIL import Image

# Cityscapes color → class ID mapping
COLOR_MAP = {
    (128, 64,128): 0,  # road
    (244, 35,232): 1,  # sidewalk
    (70, 70, 70): 2,   # building
    (102,102,156): 3,  # wall
    (190,153,153): 4,  # fence
    (153,153,153): 5,  # pole
    (250,170, 30): 6,  # traffic light
    (220,220,  0): 7,  # traffic sign
    (107,142, 35): 8,  # vegetation
    (152,251,152): 9,  # terrain
    (70,130,180): 10,  # sky
    (220, 20, 60): 11, # person
    (255,  0,  0): 12, # rider
    (0,  0,142): 13,   # car
    (0,  0, 70): 14,   # truck
    (0, 60,100): 15,   # bus
    (0, 80,100): 16,   # train
    (0,  0,230): 17,   # motorcycle
    (119, 11, 32): 18, # bicycle
}

def convert_color_to_id(img, tol=15):
    """Convert a color mask to ID mask with small color tolerance."""
    arr = np.array(img).astype(np.int16)
    h, w, _ = arr.shape
    id_mask = np.zeros((h, w), dtype=np.uint8)
    for color, idx in COLOR_MAP.items():
        color_arr = np.array(color, dtype=np.int16).reshape(1,1,3)
        dist = np.linalg.norm(arr - color_arr, axis=-1)
        matches = dist <= tol
        id_mask[matches] = idx
    return Image.fromarray(id_mask)

def convert_folder(input_dir, output_dir, tol=15):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg'))])
    for fname in files:
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        mask = convert_color_to_id(img, tol=tol)
        mask.save(os.path.join(output_dir, fname))
        print("Converted:", fname)

if __name__ == "__main__":
    convert_folder(r"E:\CityScape_Dataset\train\label", r"E:\CityScape_Dataset\train\label_id", tol=15)
    convert_folder(r"E:\CityScape_Dataset\val\label",   r"E:\CityScape_Dataset\val\label_id",   tol=15)
