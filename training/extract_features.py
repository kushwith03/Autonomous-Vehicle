import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import CarlaDataset
from models.autoencoder import AutoEncoder
from utils.helpers import get_device, load_config

def extract_features(config_path, checkpoint_path, labels_csv, output_csv):
    """Encodes images into latent features and saves as CSV for Stage 2 training."""
    cfg = load_config(config_path)
    device = get_device(cfg['device'])
    
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg['models']['autoencoder']['img_size'])),
        transforms.ToTensor(),
    ])
    
    # Load labels to match images with control signals
    print(f"Loading data from: {labels_csv}")
    if not os.path.exists(labels_csv):
        print(f"[ERROR] Labels CSV not found: {labels_csv}")
        return
        
    # Dataset needs to return path so we can match it with labels
    dataset = CarlaDataset(cfg['paths']['train_list'], cfg['paths']['data_root'], transform, labels_csv=labels_csv, return_path=True)
    if len(dataset) == 0:
        print("[ERROR] Dataset is empty.")
        return
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    latent_data = []
    
    print("Extracting features (this may take a while)...")
    with torch.no_grad():
        for imgs, labels, rel_paths in tqdm(loader):
            imgs = imgs.to(device)
            latents = model.encode(imgs).cpu().numpy()
            
            for i, rel_path in enumerate(rel_paths):
                # labels[i] already contains [steer, throttle, brake] from CarlaDataset
                row = list(latents[i]) + list(labels[i].numpy())
                latent_data.append(row)

    if not latent_data:
        print("[ERROR] No matched latent features extracted! Check dataset and labels_csv.")
        return
        
    # Define column names
    latent_cols = [f"f_{i}" for i in range(latents.shape[1])]
    control_cols = ["steer", "throttle", "brake"]
    df_output = pd.DataFrame(latent_data, columns=latent_cols + control_cols)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    print(f"Extraction complete: Saved {len(df_output)} samples to {output_csv}")

if __name__ == "__main__":
    pass
