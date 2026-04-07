# Production-Quality Autonomous Driving System for CARLA

A modular and scalable autonomous driving pipeline using a two-stage Behavioral Cloning approach.

## Overview
This system decomposes the autonomous driving task into two stages:
1. **Vision (AutoEncoder):** Learns to compress high-dimensional camera frames into a compact latent representation.
2. **Control (ControlNet):** Learns to map these latent features to vehicle control signals (Steer, Throttle, Brake).

By using latent features, the control model is more robust to noise and easier to train on limited datasets.

## Project Structure
```text
autonomous-vehicle/
├── configs/            # YAML configurations for all modules
├── data/               # Unified Dataset and preprocessing
├── models/             # PyTorch definitions for AE and Controller
├── training/           # Scripts for Training and Feature Extraction
├── inference/          # Predictor class for real-time control
├── carla_integration/  # CARLA client and sensor handling
├── utils/              # Helper functions and Logger
├── main.py             # Single entry point CLI
└── requirements.txt    # Project dependencies
```

## Setup Instructions

### 1. Requirements
- Python 3.7 or 3.8
- CARLA Simulator (0.9.13 recommended)
- GPU with CUDA support (recommended)

### 2. Installation
```bash
git clone <your-repo-url>
cd autonomous-vehicle
pip install -r requirements.txt
```

### 3. Configuration
Edit `configs/default_config.yaml` to set your:
- Data paths (`data_root`, `train_list`)
- CARLA egg path (`carla_env.egg_path`)
- Training hyperparameters

## Usage Guide

### Stage 1: Train AutoEncoder
Train the model to understand the visual environment.
```bash
python main.py --mode train_ae
```

### Stage 2a: Extract Latent Features
Generate the latent dataset using a trained AutoEncoder checkpoint.
```bash
python main.py --mode extract_features --ae_path results/checkpoints/ae_final.pth --labels_csv path/to/recorded_controls.csv
```

### Stage 2b: Train Controller
Train the driving logic using the extracted features.
```bash
python main.py --mode train_ctrl --ae_path results/checkpoints/ae_final.pth
```

### Stage 3: Autonomous Driving
Deploy the trained models in the CARLA simulator.
```bash
python main.py --mode drive --ae_path results/checkpoints/ae_final.pth --ctrl_path results/checkpoints/ctrl_final.pth
```

## Features
- Modular design for easy experimentation.
- Config-driven pipeline (no hardcoded parameters in logic).
- TensorBoard integration for training visualization.
- Robust CARLA sensor management and error handling.
- Clean, human-readable code following PEP8 standards.
```
