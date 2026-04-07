import torch
import torch.nn as nn

class ControlNet(nn.Module):
    def __init__(self, latent_dim=8192, hidden_dim=256, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh() # steer, throttle, brake in range [-1, 1]
        )

    def forward(self, z):
        return self.net(z)
