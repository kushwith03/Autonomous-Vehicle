# File: E:\AutonomousVehicle\convert_to_kitti.py
import os
import shutil

# Source folder (your CARLA raw images)
SOURCE_DIR = r"E:\AutonomousVehicle\datasets\carla_raw"

# KITTI-like output folder
OUTPUT_DIR = r"E:\AutonomousVehicle\datasets\kitti_converted"

# Create KITTI structure
subfolders = [
    "image_2",      # RGB images
    "label_2",      # labels (empty for now)
    "calib",        # calibration files
    "velodyne",     # lidar data (empty)
]

for sf in subfolders:
    os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)

# Copy RGB images to image_2
image_target = os.path.join(OUTPUT_DIR, "image_2")

print("Copying RGB images...")
for i, filename in enumerate(sorted(os.listdir(SOURCE_DIR))):
    if filename.endswith(".png"):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(image_target, f"{i:06d}.png")
        shutil.copy(src, dst)

print("✅ Conversion complete!")
print(f"Total images copied: {len(os.listdir(image_target))}")
print(f"KITTI dataset created at: {OUTPUT_DIR}")
