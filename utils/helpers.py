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

def setup_carla_path(egg_path):
    """Adds the CARLA egg to sys.path if specified in config."""
    if egg_path and os.path.exists(egg_path):
        if egg_path not in sys.path:
            sys.path.append(egg_path)
            print(f"[INFO] Added CARLA egg to sys.path: {egg_path}")
    else:
        print("[WARN] CARLA egg path not found. Ensure carla is installed in your environment.")
on as fallback
        standard_path = r"C:\CARLA_0.9.13\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg"
        if os.path.exists(standard_path) and standard_path not in sys.path:
            sys.path.append(standard_path)
