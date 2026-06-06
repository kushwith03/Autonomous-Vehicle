# Autonomous Driving Pipeline for CARLA

A production-grade autonomous vehicle control system featuring a decoupled perception-control architecture and behavioral cloning logic.

## Overview

This project implements a robust **Two-Stage Behavioral Cloning** pipeline designed for the CARLA simulator. By separating visual perception from vehicle control, the system achieves better generalization and safety compared to traditional end-to-end models.

### Architecture
1. **Perception (AutoEncoder):** A Convolutional AutoEncoder compresses $128 \times 128 \times 3$ RGB images into an $8192$-dimensional latent vector, forcing the model to learn semantic representations of the road.
2. **Control (ControlNet):** A Multi-Layer Perceptron (MLP) maps these latent features directly to vehicle control signals (Steer, Throttle, Brake).

- **GitHub Repository:** [github.com/kushwith03/Autonomous-Vehicle](https://github.com/kushwith03/Autonomous-Vehicle)

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Simulation:** CARLA Simulator (0.9.13+)
- **Computer Vision:** OpenCV, PIL
- **Analytics:** NumPy, Pandas, Scikit-learn, TensorBoard

## Key Engineering Features

- **Decoupled Design:** High explainability by separating visual understanding from control logic.
- **Latency Tracking:** Real-time monitoring achieving **sub-50ms inference latency** for real-time safety.
- **Behavioral Cloning:** Trained using CNN-based models to predict steering and throttle control from human driver data.
- **Validation Loops:** Automated validation during training stages to prevent overfitting and ensure precision.

## Project Structure

```text
Autonomous-Vehicle/
├── carla_integration/  # Simulation client and weather management
├── models/             # AE, ControlNet, and CNN architectures
├── training/           # Modular training scripts and validation
├── inference/          # Real-time predictor with performance tracking
├── configs/            # Centralized YAML hyperparameter control
└── utils/              # Latency benchmarks and navigation metrics
```

## Performance Target
- **Inference Latency:** $\le 50\text{ms}$
- **Navigation Precision:** Optimized steering and throttle control across complex simulated environments.

## Author

**R Khushwith Kumar**  
Software Engineer | Deep Learning Enthusiast  
[Portfolio](https://rkhushwith-portfolio.vercel.app) • [GitHub](https://github.com/kushwith03) • [LinkedIn](https://linkedin.com/in/kushwith03)
