# train_segmentation.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from tqdm import tqdm  # ✅ for progress bar
from dataset import CityscapesDataset


def collate_fn(batch):
    """Custom collate function for variable-sized masks."""
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.from_numpy(np.stack([b[1] for b in batch])).long()
    fnames = [b[2] for b in batch]
    return imgs, masks, fnames


def main(args):
    device = torch.device("cpu")
    print("✅ Using device:", device)

    # -------------------------
    # Load datasets
    # -------------------------
    print("🔹 Loading dataset...")
  # dataset init: change size to (128,256)
    train_ds = CityscapesDataset(args.train_images, args.train_masks, size=(256, 512), augment=True)
    val_ds   = CityscapesDataset(args.val_images, args.val_masks, size=(256, 512), augment=False)


    # model
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)

    print(f"🔹 Training samples: {len(train_ds)}")
    print(f"🔹 Validation samples: {len(val_ds)}")

    # -------------------------
    # DataLoaders (Windows-safe + skip last incomplete batch)
    # -------------------------
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True  # ✅ skip incomplete batch to prevent BatchNorm errors
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True
    )

    # -------------------------
    # Model setup
    # -------------------------
    print("🔹 Initializing DeepLabV3 model...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, args.num_classes, kernel_size=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    best_val_loss = float("inf")

    # -------------------------
    # Resume training if checkpoint exists
    # -------------------------
    checkpoint_path = os.path.join(args.save_dir, "deeplabv3_cityscapes_cpu.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"🔁 Resumed training from checkpoint: {checkpoint_path}")

    # -------------------------
    # Training loop
    # -------------------------
    print("✅ Starting training...\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        print(f"🚀 Epoch {epoch + 1}/{args.epochs} - Training...")
        train_progress = tqdm(train_dl, desc=f"Training Epoch {epoch + 1}", ncols=100)
        for imgs, masks, _ in train_progress:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dl)
        print(f"\n[Epoch {epoch + 1}/{args.epochs}] 🔸 Train Loss: {avg_train_loss:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        print(f"🔍 Epoch {epoch + 1}/{args.epochs} - Validating...")
        val_progress = tqdm(val_dl, desc=f"Validating Epoch {epoch + 1}", ncols=100)
        with torch.no_grad():
            for imgs, masks, _ in val_progress:
                imgs = imgs.to(device)
                masks = masks.to(device)

                outputs = model(imgs)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_dl)
        print(f"🔹 Validation Loss: {avg_val_loss:.4f}")

        # -------------------------
        # Save best model
        # -------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model: {checkpoint_path}\n")

    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on Cityscapes")
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_masks", required=True)
    parser.add_argument("--val_images", required=True)
    parser.add_argument("--val_masks", required=True)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=19)
    args = parser.parse_args()

    main(args)

