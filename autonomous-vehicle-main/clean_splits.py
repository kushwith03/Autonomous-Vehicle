# File: E:\AutonomousVehicle\clean_splits.py
import os, shutil

COMBINED_ROOT = r"E:\AutonomousVehicle\datasets\combined"
IMG_DIR = os.path.join(COMBINED_ROOT, "image_2")
FILES = ["train.txt", "val.txt"]

for fname in FILES:
    src = os.path.join(COMBINED_ROOT, fname)
    if not os.path.isfile(src):
        print("Missing:", src)
        continue
    backup = src + ".bak"
    shutil.copy(src, backup)
    kept = []
    removed = []
    with open(src, "r") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            # convert relative path to actual path
            path = os.path.join(COMBINED_ROOT, rel.replace("/", os.sep))
            if os.path.exists(path):
                kept.append(rel)
            else:
                removed.append(rel)
    # write cleaned file
    cleaned = os.path.join(COMBINED_ROOT, fname)
    with open(cleaned, "w") as f:
        for r in kept:
            f.write(r + "\n")
    print(f"{fname}: kept {len(kept)} entries, removed {len(removed)} (backup at {backup})")
    if removed:
        print(" Example removed:", removed[:5])
print("Done.")
