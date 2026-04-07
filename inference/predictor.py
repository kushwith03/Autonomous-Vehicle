import torch
import numpy as np
from torchvision import transforms
from models.autoencoder import AutoEncoder
from models.controller import ControlNet

class Predictor:
    def __init__(self, ae_path, ctrl_path, device, config):
        self.device = device
        self.config = config
        
        self.ae = AutoEncoder().to(device)
        self.ae.load_state_dict(torch.load(ae_path, map_location=device))
        self.ae.eval()
        
        latent_dim = config['models']['autoencoder']['latent_dim']
        hidden_dim = config['models']['controller']['hidden_dim']
        output_dim = config['models']['controller']['output_dim']
        
        self.ctrl = ControlNet(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        self.ctrl.load_state_dict(torch.load(ctrl_path, map_location=device))
        self.ctrl.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(tuple(config['models']['autoencoder']['img_size'])),
            transforms.ToTensor(),
        ])
        print("Models loaded and Predictor ready.")

    def predict(self, image_array):
        with torch.no_grad():
            tensor = self.transform(image_array).unsqueeze(0).to(self.device)
            latent = self.ae.encode(tensor)
            actions = self.ctrl(latent).cpu().numpy().flatten()
        
        steer, throttle, brake = actions
        
        # Scaling and clipping (assuming Tanh was removed and model trained on raw values)
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        
        # Simple heuristic: if braking, reduce throttle
        if brake > 0.1:
            throttle = 0.0
            
        return steer, throttle, brake
