import torch
import torch.nn as nn

class CNNController(nn.Module):
    """
    End-to-end CNN for behavioral cloning. Maps raw camera frames (3x128x128) 
    directly to vehicle control signals (steer, throttle, brake).
    """
    def __init__(self, output_dim=3, dropout=0.3):
        super(CNNController, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1: (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv block 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv block 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv block 4: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((4, 4)) # -> (256, 4, 4)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        """
        Forward pass for the CNN controller.
        Input: (B, 3, 128, 128) image tensor.
        Output: (B, 3) logits for (steer, throttle, brake).
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
