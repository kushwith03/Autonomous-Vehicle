import os
import pandas as pd
import numpy as np
import torch
from models.controller import ControlNet
from utils.metrics import compute_navigation_error, compute_improvement_pct
from utils.helpers import load_config

def evaluate_controller(config_path, latent_csv, checkpoint_path):
    """
    Load ControlNet from checkpoint
    Load latent CSV, split 80/20
    Run inference on val split
    Compute metrics using compute_navigation_error
    Compute improvement vs a zero-prediction baseline
    Print formatted report including % improvement over baseline
    Return metrics dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating Controller on: {device}")

    # Load data
    df = pd.read_csv(latent_csv)
    # Assume CSV has 'latent' (8192 space separated values or similar) and 'steer', 'throttle', 'brake'
    # For simplicity, if 'latent' is a column, we split it.
    
    # Check if 'latent' column exists
    if 'latent' in df.columns:
        X = np.array([list(map(float, x.split())) for x in df['latent']])
    else:
        # Fallback: assume all columns except labels are latent features
        X = df.drop(['steer', 'throttle', 'brake'], axis=1).values
        
    y = df[['steer', 'throttle', 'brake']].values
    
    # 80/20 split
    split_idx = int(0.8 * len(df))
    # X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    X_val = X[split_idx:]

    # Load model
    input_dim = X.shape[1]
    model = ControlNet(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Inference
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_val_tensor).cpu().numpy()

    # Compute metrics
    metrics = compute_navigation_error(y_val, y_pred)
    
    # Baseline: predict all zeros
    y_baseline = np.zeros_like(y_val)
    # Wait, usually throttle is 0.5 for moving forward, but Task 4 says "predicts all zeros"
    # "predict all zeros, compute MAE -> this is the before baseline"
    baseline_metrics = compute_navigation_error(y_val, y_baseline)
    
    pct_improvement = compute_improvement_pct(baseline_metrics['overall_mae'], metrics['overall_mae'])

    print("\n" + "="*40)
    print("EVALUATION REPORT")
    print("-" * 40)
    print(f"Val Samples: {len(y_val)}")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("-" * 40)
    print(f"Baseline MAE: {baseline_metrics['overall_mae']:.4f}")
    print(f"Navigation Error Reduction vs Zero Baseline: {pct_improvement:.1f}%")
    print("="*40 + "\n")

    return metrics

if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--latent_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate_controller(args.config, args.latent_csv, args.checkpoint)
