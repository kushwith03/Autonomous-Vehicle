# Modular Autonomous Driving Pipeline for CARLA

An interview-ready, production-quality autonomous vehicle control system featuring a decoupled perception-control architecture.

## 🚀 Overview
This project implements a robust **Two-Stage Behavioral Cloning** pipeline designed for the CARLA simulator. By separating visual perception from vehicle control, the system achieves better generalization and more efficient training compared to traditional end-to-end models.

### The Two-Stage Architecture
1.  **Perception (AutoEncoder):** A Convolutional AutoEncoder compresses $128 \times 128 \times 3$ RGB images into an $8192$-dimensional latent vector. This forces the model to learn a compact, semantic representation of the driving environment.
2.  **Control (ControlNet):** A Multi-Layer Perceptron (MLP) maps these latent features directly to vehicle control signals (Steer, Throttle, Brake).

**Why this approach?**
- **Data Efficiency:** The perception module can be pre-trained on large unlabeled datasets.
- **Explainability:** It is easier to debug whether a failure occurred in "seeing" the road or "deciding" how to drive.
- **Robustness:** Latent features are less sensitive to pixel-level noise compared to raw image inputs.

---

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Simulation:** CARLA Simulator (0.9.13+)
- **Computer Vision:** OpenCV, PIL
- **Data Handling:** NumPy, Pandas, Scikit-learn
- **Monitoring:** TensorBoard, TQDM

---

## 📂 Project Structure
```text
Autonomous-Vehicle/
├── carla_integration/  # CARLA client, weather, and sensor management
├── configs/            # Centralized YAML configuration
├── data/               # Stable Dataset classes (handling images & labels)
├── inference/          # Real-time Predictor with latency tracking
├── models/             # AE, ControlNet, and CNNController (baseline)
├── training/           # Modular training scripts with validation loops
├── utils/              # Latency benchmarks, metrics, and helpers
├── tests/              # Unit tests for models and navigation metrics
├── main.py             # Unified CLI entry point
└── requirements.txt    # Project dependencies
```

---

## 🚦 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training Pipeline

**Stage 1: Perception**
```bash
python main.py --mode train_ae
```

**Stage 2a: Feature Extraction**
Pre-calculate latents to speed up controller training by 10x.
```bash
python main.py --mode extract_features --ae_path results/checkpoints/autoencoder_best.pth --labels_csv datasets/carla_labels.csv
```

**Stage 2b: Control**
```bash
python main.py --mode train_ctrl --latent_csv results/latent_features.csv
```

### 3. Evaluation & Benchmarking
```bash
# Evaluate model accuracy vs baseline
python main.py --mode evaluate --ctrl_path results/checkpoints/controller_best.pth --latent_csv results/latent_features.csv

# Benchmark inference latency
# Target: < 50ms (20+ FPS)
python main.py --mode drive --ae_path ... --ctrl_path ... # or dedicated benchmark mode if added
```

### 4. Autonomous Driving
```bash
python main.py --mode drive --ae_path results/checkpoints/autoencoder_best.pth --ctrl_path results/checkpoints/controller_best.pth
```

---

## ✨ Key Engineering Features
- **Validation Loops:** All training stages include automated validation to prevent overfitting.
- **Latency Tracking:** Real-time monitoring of inference speed (ms) to ensure simulation stability.
- **Type-Safe Data Loading:** `CarlaDataset` ensures consistent output formats, preventing runtime crashes during training.
- **Decoupled Configuration:** Full control over hyperparameters via `configs/default_config.yaml` without touching code.
- **Unit Tested:** Includes tests for model architectures and navigation metrics (MAE, RMSE).
- **CI/CD Ready:** GitHub Actions workflow included for automated linting and testing.

---

## 📊 Performance Metrics
The system is evaluated on:
- **MAE (Mean Absolute Error):** For steering, throttle, and brake accuracy.
- **Improvement over Baseline:** Performance compared to a "Zero-Control" or "Mean-Control" model.
- **Inference Latency:** Target is $\le 50\text{ms}$ for real-time safety.
