import time
import torch
from torchvision import transforms
from PIL import Image

class Predictor:
    def __init__(self, ae_path, ctrl_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models (placeholder for actual model instantiation if they are different)
        # For benchmarking, we need to know what models are being used.
        # Task 3 says "instantiate Predictor", and "require --ae_path and --ctrl_path"
        
        from models.autoencoder import AutoEncoder
        # Assuming ctrl_path refers to an MLP controller that takes latent features
        # Task 2b in README mentions: BC Controller (MLP) -> Latent features -> (steer, throttle, brake)
        # Let's create models/controller.py first as it's implied.
        
        self.ae = AutoEncoder().to(self.device)
        self.ae.load_state_dict(torch.load(ae_path, map_location=self.device))
        self.ae.eval()
        
        # We need the MLP Controller as well.
        from models.controller import ControlNet
        self.ctrl = ControlNet(8192, 256, 3).to(self.device)
        self.ctrl.load_state_dict(torch.load(ctrl_path, map_location=self.device))
        self.ctrl.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        self.last_latency_ms = 0.0

    def predict(self, image):
        """
        image: PIL Image or numpy array
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            img_tensor = image.to(self.device)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
        else:
            # assume numpy
            image = Image.fromarray(image)
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            latent = self.ae.encode(img_tensor)
            control = self.ctrl(latent)
        t1 = time.perf_counter()
        
        self.last_latency_ms = (t1 - t0) * 1000
        return control.cpu().numpy()[0]
