# check.py
import os
import numpy as np
from PIL import Image

# Path to your converted label_id folder
path = r"E:\CityScape_Dataset\train\label_id"

files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))])

for fname in files[:5]:
    mask = np.array(Image.open(os.path.join(path, fname)))
    print(fname, "unique values:", np.unique(mask))
