import torch
import yaml
import os
import sys

def get_device(preference="auto"):
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dirs(dirs):
    for d in dirs:
        if d: os.makedirs(d, exist_ok=True)

def setup_carla_env(cfg):
    """Adds the CARLA egg to sys.path if specified in config."""
    egg_path = cfg.get('carla_env', {}).get('egg_path', '')
    if egg_path and os.path.exists(egg_path):
        if egg_path not in sys.path:
            sys.path.append(egg_path)
            print(f"[INFO] Added CARLA egg to sys.path: {egg_path}")
    else:
        print("[WARN] CARLA egg path not found in config or doesn't exist.")
