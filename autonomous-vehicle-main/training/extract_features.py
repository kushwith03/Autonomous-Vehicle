import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from data.carla_dataset import CarlaDataset
from models.autoencoder import AutoEncoder
from tqdm import tqdm

def extract_features(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Extracting Features using: {device}")

    transform = transforms.Compose([
        transforms.Resize(tuple(cfg['models']['autoencoder']['input_dim'][1:])),
        transforms.ToTensor(),
    ])

    dataset = CarlaDataset(cfg['paths']['carla_root'], labels_csv=cfg['paths']['carla_labels_csv'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ae = AutoEncoder().to(device)
    ae_path = os.path.join(cfg['paths']['checkpoints'], "latest_autoencoder.pth")
    if not os.path.exists(ae_path):
        print(f"Error: AutoEncoder checkpoint not found at {ae_path}")
        return
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    features = []
    for imgs, labels in tqdm(dataloader):
        imgs = imgs.to(device)
        with torch.no_grad():
            latent = ae.encode(imgs).cpu().numpy()[0]
        
        feature_str = " ".join(map(str, latent))
        features.append({
            "latent": feature_str,
            "steer": labels[0][0].item(),
            "throttle": labels[0][1].item(),
            "brake": labels[0][2].item()
        })

    df = pd.DataFrame(features)
    output_path = "datasets/val_latents.csv"
    os.makedirs("datasets", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Extracted features saved to {output_path}")

if __name__ == "__main__":
    extract_features("configs/default_config.yaml")
