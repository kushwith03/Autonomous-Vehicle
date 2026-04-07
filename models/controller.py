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
            nn.Linear(hidden_dim // 2, output_dim)
            # Removed Tanh to allow targets in native [0, 1] for throttle/brake and [-1, 1] for steer
        )

    def forward(self, z):
        return self.net(z)
