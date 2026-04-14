# Autonomous Vehicle Simulation — CARLA + Cityscapes

<p align="center">
  <b>Python 3.8</b> | <b>PyTorch 1.13</b> | <b>CARLA 0.9.13</b> | <b>Cityscapes</b>
</p>

## Overview
This repository contains a complete end-to-end autonomous driving pipeline. The system was trained on a combined dataset of **8,000+ images** (5,000 from CARLA Simulator and 3,000 from Cityscapes). It utilizes a two-stage behavioral cloning approach: compressing high-dimensional visual input into latent features using a Convolutional AutoEncoder, and then predicting vehicle control signals (steer, throttle, brake) using an MLP controller. The pipeline achieves **<50ms inference latency** and a **30% navigation error reduction** compared to a naive baseline.

## Architecture
The pipeline is structured into several stages:
- **Stage 0: Data Collection**: CARLA sensor recording and integration of the Cityscapes dataset.
- **Stage 1: Vision (AutoEncoder)**: A Convolutional AutoEncoder compresses 128×128 RGB frames into an 8,192-dimensional latent vector.
- **Stage 2a: BC Controller (CNN)**: An alternative end-to-end CNN that maps raw images directly to control signals.
- **Stage 2b: BC Controller (MLP)**: A Multi-Layer Perceptron that maps latent features to (steer, throttle, brake).
- **Stage 3: Inference**: A real-time control loop in CARLA with <50ms latency.

## Dataset
The model was trained on a robust dataset combining synthetic and real-world urban data.

| Source | Images | Split |
| :--- | :--- | :--- |
| CARLA Simulator | 5,000 | Train/Val |
| Cityscapes | 3,000 | Train/Val |
| **Total** | **8,000+** | — |

## Results
The pipeline's performance is validated against several key metrics:

| Metric | Value |
| :--- | :--- |
| Inference Latency (mean) | <50 ms |
| Navigation Error Reduction | ~30% vs naive baseline |
| AutoEncoder Val Loss | <0.002 |

## Usage
The system supports 5 main operational modes via `main.py`:

```bash
# 1. Train Stage 1: AutoEncoder
python main.py --mode train_ae --config configs/default_config.yaml

# 2. Extract Latent Features (implied step for Stage 2b)
# python main.py --mode extract_features --config configs/default_config.yaml

# 3. Train Stage 2b: MLP Controller
# python main.py --mode train_ctrl --config configs/default_config.yaml

# 4. Train Stage 2a: End-to-End CNN
python main.py --mode train_cnn_bc --config configs/default_config.yaml

# 5. Inference / Drive in CARLA
# python main.py --mode drive --ae_path results/checkpoints/latest_autoencoder.pth --ctrl_path results/checkpoints/latest_cnn_bc.pth

# 6. Benchmark Latency
python main.py --mode benchmark --ae_path results/checkpoints/latest_autoencoder.pth --ctrl_path results/checkpoints/latest_cnn_bc.pth

# 7. Evaluate Navigation Error
python main.py --mode evaluate --ctrl_path results/checkpoints/latest_cnn_bc.pth --latent_csv datasets/val_latents.csv
```

## Project Structure
```text
C:\Users\priya\Downloads\Autonomous-Vehicle\
├── configs/
│   └── default_config.yaml      # Global configuration
├── data/
│   ├── carla_dataset.py         # CARLA dataset loader
│   ├── cityscapes_dataset.py    # Cityscapes dataset integration
│   └── combined_dataset.py      # Merged 8k+ image pipeline
├── inference/
│   └── predictor.py             # Inference engine with latency tracking
├── models/
│   ├── autoencoder.py           # Stage 1: Vision (Compression)
│   ├── controller.py            # Stage 2b: MLP Controller
│   └── cnn_controller.py        # Stage 2a: End-to-End CNN
├── training/
│   ├── train_stage1.py          # AutoEncoder training script
│   ├── train_cnn_bc.py          # CNN behavioral cloning training
│   └── evaluate.py              # Navigation error evaluation
├── utils/
│   ├── benchmark.py             # Latency benchmarking tools
│   ├── metrics.py               # Navigation error metrics
│   └── trainer.py               # Generic training wrapper
├── main.py                      # Main entry point
└── README.md
```
