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
    df_labels = pd.read_csv(labels_csv)
    
    # Dataset needs to return path so we can match it with labels
    dataset = CarlaDataset(cfg['paths']['train_list'], cfg['paths']['data_root'], transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    latent_data = []
    
    print("Extracting features (this may take a while)...")
    with torch.no_grad():
        for imgs, rel_paths in tqdm(loader):
            imgs = imgs.to(device)
            latents = model.encode(imgs).cpu().numpy()
            
            for i, rel_path in enumerate(rel_paths):
                # Simple matching logic: find filename in labels_csv
                # We assume the CSV has a column 'image' that matches rel_path filename
                filename = os.path.basename(rel_path)
                
                # Try to find matching row in labels
                # Note: This logic depends on the specific dataset recording format
                match = df_labels[df_labels['image'].str.contains(filename)]
                if not match.empty:
                    controls = match.iloc[0][['steer', 'throttle', 'brake']].values
                    row = list(latents[i]) + list(controls)
                    latent_data.append(row)

    # Define column names
    latent_cols = [f"f_{i}" for i in range(latents.shape[1])]
    control_cols = ["steer", "throttle", "brake"]
    df_output = pd.DataFrame(latent_data, columns=latent_cols + control_cols)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    print(f"Extraction complete: Saved {len(df_output)} samples to {output_csv}")

if __name__ == "__main__":
    # Can be run manually or via main.py
    pass
