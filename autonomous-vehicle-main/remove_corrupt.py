# File: E:\AutonomousVehicle\remove_corrupt.py
# Scans E:\AutonomousVehicle\datasets\combined\image_2 for unreadable images.
# Moves unreadable/corrupted files to a folder: E:\AutonomousVehicle\datasets\combined\corrupt

import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False  # we want to detect truly truncated files

SRC = r"E:\AutonomousVehicle\datasets\combined\image_2"
DST_DIR = r"E:\AutonomousVehicle\datasets\combined\corrupt"
os.makedirs(DST_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(SRC) if f.lower().endswith(('.png','.jpg','.jpeg'))])
total = len(files)
moved = 0

print(f"Scanning {total} images in:", SRC)
for i, fname in enumerate(files, 1):
    path = os.path.join(SRC, fname)
    try:
        with Image.open(path) as im:
            im.verify()   # verify detects many corrupted/truncated files
        # re-open to ensure full decode for some edge cases
        with Image.open(path) as im:
            im.load()
    except Exception as e:
        print(f"[{i}/{total}] Corrupt - moving: {fname}  -> {e}")
        try:
            os.replace(path, os.path.join(DST_DIR, fname))
        except Exception as ee:
            print("  Failed to move:", ee)
        moved += 1
    else:
        if i % 500 == 0:
            print(f"[{i}/{total}] OK so far...")

print("Done.")
print("Total checked:", total)
print("Moved (corrupt) files:", moved)
print("Corrupted files moved to:", DST_DIR)
