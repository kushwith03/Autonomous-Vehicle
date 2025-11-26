# Autonomous Vehicle Simulation using CARLA

This project implements an end-to-end autonomous driving pipeline using the CARLA Simulator.  
The system performs data collection, semantic segmentation, decision making, and real-time vehicle control inside the simulator.

---

## Project Overview

The project consists of four major components:

1. Data Collection from the CARLA simulator  
2. Semantic Segmentation Model (PyTorch)  
3. Decision-Making Model (Throttle, Steering, Brake)  
4. Full Integration: real-time autonomous driving inside CARLA  

The goal is to build a minimal autonomous driving stack capable of perceiving the scene and controlling the vehicle.

---

## Features

- Automated dataset collection (RGB images, segmentation masks, expert driving commands)
- Custom PyTorch segmentation model (U-Net)
- Decision model trained on expert control data
- Integrated inference pipeline for real-time driving
- Utilities for dataset cleaning and preprocessing

---

## Folder Structure

```
AutonomousVehicle/
│
├── carla_scripts/
│ ├── spawn_vehicle.py
│ ├── collect_data.py
│ ├── integrate_model.py
│ ├── test_connection.py
│
├── segmentation/
│ ├── dataset.py
│ ├── train_segmentation.py
│ ├── infer_segmentation.py
│ ├── verify_dataset.py
│ ├── convert_masks.py
│
├── decision_model/
│ ├── train_decision_model.py
│ ├── inference_local.py
│
├── utils/
│ ├── merge_and_resize.py
│ ├── remove_corrupt.py
│
├── demo/
│ ├── sample_rgb.png
│ ├── sample_mask.png
│ ├── sample_colored_mask.png
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Requirements

- Python 3.7+
- CARLA 0.9.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## Results

- Semantic segmentation trained using custom CARLA dataset  
- Decision model predicts steering/throttle/brake values  
- Integrated system performs autonomous lane following inside CARLA  

(Demo images/videos can be added in the `demo/` folder)

---

## Future Work

- Improve segmentation using DeepLab / transformer-based models  
- Add Lidar + sensor fusion  
- Implement waypoint navigation  
- Reinforcement learning for improved control  

---

