# File: E:\AutonomousVehicle\train_test.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ImageFolderDataset
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
root_dir = r"E:\AutonomousVehicle\datasets\combined"
train_list = os.path.join(root_dir, "train.txt")
results_dir = r"E:\AutonomousVehicle\results"
os.makedirs(results_dir, exist_ok=True)

epochs = 10
batch_size = 16
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Training on:", device)

# ---------------- MODEL ----------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------------- DATA LOADER ----------------
def make_dataloader():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Matches inference size
        transforms.ToTensor(),
    ])
    train_ds = ImageFolderDataset(train_list, root_dir, transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_dl

# ---------------- TRAINING LOOP ----------------
def train_loop():
    train_dl = make_dataloader()
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (imgs, _) in enumerate(train_dl):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_dl)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_dl))
        print(f"✅ Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    print("✅ Training complete.")
    return model, train_dl

# ---------------- VISUALIZATION & SAVE ----------------
def visualize_and_save(model, train_dl):
    save_path = os.path.join(results_dir, "model_autoencoder.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved at: {save_path}")

    try:
        imgs, _ = next(iter(train_dl))
    except StopIteration:
        print("⚠️ No images to visualize.")
        return

    model.eval()
    with torch.no_grad():
        outputs = model(imgs.to(device)).cpu()

    for i in range(min(3, imgs.shape[0])):
        orig = imgs[i].permute(1, 2, 0).numpy()
        recon = outputs[i].permute(1, 2, 0).numpy()
        orig = np.clip(orig, 0.0, 1.0)
        recon = np.clip(recon, 0.0, 1.0)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[1].imshow(recon)
        axes[1].set_title("Reconstructed")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_file = os.path.join(results_dir, f"recon_{i}.png")
        plt.savefig(out_file)
        plt.close(fig)
        print("🖼️ Saved reconstruction:", out_file)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    model, train_dl = train_loop()
    visualize_and_save(model, train_dl)
