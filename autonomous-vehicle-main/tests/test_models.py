import torch
import sys
import os

# Add the project root to sys.path to allow imports from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import AutoEncoder
from models.controller import ControlNet
from models.cnn_controller import CNNController

def test_autoencoder_forward():
    model = AutoEncoder()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)

def test_autoencoder_encode():
    model = AutoEncoder()
    x = torch.randn(2, 3, 128, 128)
    latent = model.encode(x)
    assert latent.shape == (2, 8192)

def test_controlnet_forward():
    model = ControlNet(8192, 256, 3)
    x = torch.randn(2, 8192)
    y = model(x)
    assert y.shape == (2, 3)

def test_cnn_controller_forward():
    model = CNNController()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3)
