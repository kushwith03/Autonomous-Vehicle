# File: E:\AutonomousVehicle\merge_resume.py
"""
Resumable merge & resize:
 - Reads CARLA images from: E:\AutonomousVehicle\datasets\kitti_converted\image_2
 - Reads Cityscapes images from: E:\AutonomousVehicle\datasets\cityscapes\leftImg8bit
 - Writes resized images (800x600) to: E:\AutonomousVehicle\datasets\combined\image_2
Usage:
  python E:\AutonomousVehicle\merge_resume.py          # resume/continue
  python E:\AutonomousVehicle\merge_resume.py --force-clean   # delete combined folder and start fresh
"""
import os
import shutil
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONFIG (edit only if you moved folders)
CARLA_IMG_DIR = r"E:\AutonomousVehicle\datasets\kitti_converted\image_2"
CITYSCAPES_DIR = r"E:\AutonomousVehicle\datasets\cityscapes\leftImg8bit"
OUTPUT_DIR = r"E:\AutonomousVehicle\datasets\combined\image_2"
TARGET_SIZE = (800, 600)
PROGRESS_EVERY = 100

def collect_cityscapes_images(city_root):
    # Cityscapes structure: leftImg8bit/train/<city>/*.png, leftImg8bit/val/...
    imgs = []
    for root, _, files in os.walk(city_root):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(root, f))
    return imgs

def safe_process_and_save(src_path, dst_path):
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            if im.size != TARGET_SIZE:
                im = im.resize(TARGET_SIZE, Image.BILINEAR)
            im.save(dst_path)
        return True
    except Exception as e:
        print(f"  [!] Skipped (corrupt?): {src_path}  -> {e}")
        return False

def main(force_clean=False):
    if force_clean and os.path.isdir(OUTPUT_DIR):
        print("Removing existing combined folder (force-clean)...")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine starting index (skip existing files)
    existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if existing:
        last = existing[-1]
        start_idx = int(os.path.splitext(last)[0])
        start_idx += 1
    else:
        start_idx = 0

    print("Starting merge/resume.")
    print("CARLA source:", CARLA_IMG_DIR)
    print("Cityscapes source:", CITYSCAPES_DIR)
    print("Output folder:", OUTPUT_DIR)
    print("Starting index:", start_idx)

    total_processed = 0
    idx = start_idx

    # 1) CARLA images (if present)
    if os.path.isdir(CARLA_IMG_DIR):
        carla_list = sorted([p for p in os.listdir(CARLA_IMG_DIR) if p.lower().endswith((".png", ".jpg", ".jpeg"))])
        for name in carla_list:
            src = os.path.join(CARLA_IMG_DIR, name)
            dst = os.path.join(OUTPUT_DIR, f"{idx:06d}.png")
            if os.path.exists(dst):
                idx += 1
                continue
            ok = safe_process_and_save(src, dst)
            if ok:
                total_processed += 1
                idx += 1
                if total_processed % PROGRESS_EVERY == 0:
                    print(f"  Processed {total_processed} images so far (last idx {idx-1})")
    else:
        print("[!] CARLA image directory not found:", CARLA_IMG_DIR)

    # 2) Cityscapes (train + val)
    if os.path.isdir(CITYSCAPES_DIR):
        cs_images = collect_cityscapes_images(CITYSCAPES_DIR)
        for src in cs_images:
            dst = os.path.join(OUTPUT_DIR, f"{idx:06d}.png")
            if os.path.exists(dst):
                idx += 1
                continue
            ok = safe_process_and_save(src, dst)
            if ok:
                total_processed += 1
                idx += 1
                if total_processed % PROGRESS_EVERY == 0:
                    print(f"  Processed {total_processed} images so far (last idx {idx-1})")
    else:
        print("[!] Cityscapes directory not found:", CITYSCAPES_DIR)

    final_count = len([x for x in os.listdir(OUTPUT_DIR) if x.lower().endswith('.png')])
    print("✅ Finished. Total images now in combined folder:", final_count)
    print("Combined images saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-clean", action="store_true", help="Delete combined folder and start fresh")
    args = parser.parse_args()
    main(force_clean=args.force_clean)
