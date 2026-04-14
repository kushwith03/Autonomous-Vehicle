import torch
import torch.nn as nn

class ControlNet(nn.Module):
    """
    MLP-based controller for behavioral cloning. 
    Maps high-dimensional latent vectors to vehicle control signals.
    """
    def __init__(self, input_dim=8192, hidden_dim=256, output_dim=3):
        super(ControlNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
