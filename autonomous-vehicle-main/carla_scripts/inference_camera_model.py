# File: E:\AutonomousVehicle\carla_scripts\inference_camera_model.py
import carla
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from torchvision import transforms

# ------------------- AutoEncoder Architecture -------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
model.load_state_dict(torch.load(
    r"E:\AutonomousVehicle\results\model_autoencoder.pth",
    map_location=device
))
model.eval()
print("✅ Model loaded successfully on", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

# ------------------- CARLA + Camera Setup -------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    print("🎥 Camera attached to vehicle successfully.")

    def process_img(image):
        # Convert raw data to numpy
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        # Preprocess for model
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)

        # Postprocess output
        output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_resized = cv2.resize((output_img * 255).astype(np.uint8), (640, 480))

        # Combine input + reconstruction
        combined = np.hstack((frame, output_resized))
        cv2.imshow("Input (Left) | AutoEncoder Output (Right)", combined)
        cv2.waitKey(1)

    camera.listen(lambda image: process_img(image))
    print("✅ Streaming camera frames through model. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🧹 Cleaning up...")
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
