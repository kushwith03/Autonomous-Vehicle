import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import yaml
import subprocess
import shutil

# Create dummy directories
os.makedirs('datasets/combined', exist_ok=True)

# Create dummy images
img_paths = []
for i in range(5):
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    img_name = f'img_{i}.png'
    img.save(f'datasets/combined/{img_name}')
    img_paths.append(img_name)

with open('datasets/combined/train.txt', 'w') as f:
    for path in img_paths:
        f.write(path + '\n')

with open('datasets/combined/val.txt', 'w') as f:
    f.write(img_paths[0] + '\n')

# Create dummy labels.csv
df = pd.DataFrame({
    'image': img_paths,
    'steer': np.random.uniform(-1, 1, 5),
    'throttle': np.random.uniform(0, 1, 5),
    'brake': np.random.uniform(0, 1, 5)
})
df.to_csv('datasets/labels.csv', index=False)

# Update config to be fast
with open('configs/default_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['training']['stage1']['epochs'] = 1
cfg['training']['stage1']['batch_size'] = 2
cfg['training']['stage2']['epochs'] = 1
cfg['training']['stage2']['batch_size'] = 2
cfg['device'] = 'cpu'

with open('configs/default_config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f)

print("Running train_ae...")
res = subprocess.run([sys.executable, 'main.py', '--mode', 'train_ae'], capture_output=True, text=True)
if res.returncode != 0:
    print("train_ae failed:")
    print(res.stderr)
    sys.exit(1)
print(res.stdout)

# find ae checkpoint
ckpts = os.listdir('results/checkpoints')
ae_ckpt = [c for c in ckpts if c.startswith('autoencoder')][0]
ae_path = os.path.join('results/checkpoints', ae_ckpt)

print("Running extract_features...")
res = subprocess.run([sys.executable, 'main.py', '--mode', 'extract_features', '--ae_path', ae_path, '--labels_csv', 'datasets/labels.csv'], capture_output=True, text=True)
if res.returncode != 0:
    print("extract_features failed:")
    print(res.stderr)
    sys.exit(1)
print(res.stdout)

print("Running train_ctrl...")
res = subprocess.run([sys.executable, 'main.py', '--mode', 'train_ctrl', '--latent_csv', 'results/latent_features.csv'], capture_output=True, text=True)
if res.returncode != 0:
    print("train_ctrl failed:")
    print(res.stderr)
    sys.exit(1)
print(res.stdout)

ckpts = os.listdir('results/checkpoints')
ctrl_ckpt = [c for c in ckpts if c.startswith('controller')][0]
ctrl_path = os.path.join('results/checkpoints', ctrl_ckpt)

print("Running mock drive...")
# We will just test predictor instantiation since drive needs CARLA
from inference.predictor import Predictor
from utils.helpers import get_device
import torch
device = get_device('cpu')
predictor = Predictor(ae_path, ctrl_path, device, cfg)
dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
steer, throttle, brake = predictor.predict(dummy_img)
print(f"Prediction: steer={steer:.2f}, throttle={throttle:.2f}, brake={brake:.2f}")

print("SUCCESS")
