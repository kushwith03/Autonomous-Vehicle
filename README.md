# End-to-End Autonomous Driving Pipeline for CARLA

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, modular autonomous driving system implemented in PyTorch for the CARLA simulator. This project employs a two-stage **Behavioral Cloning** architecture to map raw camera pixels to vehicle control signals (Steer, Throttle, Brake).

---

## 🚀 System Architecture

The pipeline decomposes the complex driving task into two specialized stages:

1.  **Stage 1: Perception (AutoEncoder)**
    *   **Goal:** Learn a compact latent representation of the visual environment.
    *   **Method:** A deep convolutional AutoEncoder compresses 128x128 RGB frames into a high-dimensional latent vector. This filters out visual noise and focuses the downstream model on essential spatial features.

2.  **Stage 2: Control (ControlNet)**
    *   **Goal:** Map latent features to driving actions.
    *   **Method:** A multi-layer perceptron (MLP) takes the latent vectors and predicts continuous control values. By training on latent features rather than raw pixels, the controller is significantly more efficient and robust to overfitting.

---

## 📂 Project Structure

```text
autonomous-vehicle/
├── carla_integration/  # CARLA client, weather control, and sensor management
├── configs/            # Centralized YAML configuration for all modules
├── data/               # Unified PyTorch Dataset with safe resource handling
├── inference/          # Real-time Predictor with sensor-to-model transformations
├── models/             # PyTorch definitions for AutoEncoder and ControlNet
├── training/           # Distributed scripts for AE training and feature extraction
├── utils/              # System-wide helpers (CARLA env setup, optimized loggers)
├── main.py             # Single entry-point CLI for the entire pipeline
└── requirements.txt    # Managed project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **CARLA Simulator:** Version 0.9.13 (recommended).
*   **Python:** 3.7 or 3.8.
*   **Hardware:** NVIDIA GPU with CUDA support is highly recommended for real-time inference.

### 2. Setup Environment
```bash
git clone https://github.com/your-username/autonomous-vehicle.git
cd autonomous-vehicle
pip install -r requirements.txt
```

### 3. Configuration
Configure your local environment in `configs/default_config.yaml`:
*   Update `carla_env.egg_path` to point to your CARLA PythonAPI egg file.
*   Set your dataset paths and training hyperparameters.

---

## 📖 Usage Guide

The system is managed through a single CLI: `main.py`.

### Step 1: Train Perception (Stage 1)
Train the AutoEncoder to compress visual data.
```bash
python main.py --mode train_ae
```

### Step 2: Feature Extraction
Generate the latent feature dataset using a trained AutoEncoder checkpoint.
```bash
python main.py --mode extract_features --ae_path results/checkpoints/ae_final.pth --labels_csv data/recorded_labels.csv
```

### Step 3: Train Controller (Stage 2)
Train the driving logic using the latent features generated in the previous step.
```bash
python main.py --mode train_ctrl --latent_csv results/latent_features.csv
```

### Step 4: Autonomous Deployment
Run the full pipeline in the CARLA simulator.
```bash
python main.py --mode drive --ae_path results/checkpoints/ae_final.pth --ctrl_path results/checkpoints/ctrl_final.pth
```

---

## ✨ Key Engineering Features

*   **Optimized Performance:** Uses $O(1)$ hash mapping for feature-to-label synchronization during dataset generation.
*   **Real-time Stability:** Implements queue-draining in the sensor manager to eliminate control lag, ensuring the model always acts on the freshest frame.
*   **Memory Efficiency:** Strict use of Python context managers for image I/O and explicit CARLA actor lifecycle management.
*   **Robust Environment Setup:** Automated discovery and patching of the CARLA PythonAPI for seamless cross-platform execution.
*   **Modular & Scalable:** Decoupled architecture allows for swapping AutoEncoder variants or ControlNet architectures without breaking the pipeline.

---

