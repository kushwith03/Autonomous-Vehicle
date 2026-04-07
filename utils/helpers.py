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
        os.makedirs(d, exist_ok=True)

def setup_carla_env(cfg):
    """Adds the CARLA egg to sys.path if specified in config."""
    egg_path = cfg.get('carla_env', {}).get('egg_path', '')
    if egg_path and os.path.exists(egg_path):
        if egg_path not in sys.path:
            sys.path.append(egg_path)
            print(f"[INFO] Added CARLA egg to sys.path: {egg_path}")
    else:
        print("[WARN] CARLA egg path not found in config or doesn't exist. Trying fallback...")
        standard_path = r"C:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg"
        if os.path.exists(standard_path):
            if standard_path not in sys.path:
                sys.path.append(standard_path)
                print(f"[INFO] Added fallback CARLA egg to sys.path: {standard_path}")
        else:
            print("[ERROR] Fallback CARLA egg path also not found. Ensure CARLA is installed.")
