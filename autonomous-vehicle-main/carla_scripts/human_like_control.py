# Phase 3 — Human-Like Decision Control
# Uses encoder features from trained AutoEncoder to control the vehicle

import carla
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from torchvision import transforms
import random

# ---------------- AutoEncoder Encoder ----------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
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

    def forward(self, x):
        return self.encoder(x)

# ---------------- Load Trained Encoder ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder().to(device)
state_dict = torch.load(r"E:\AutonomousVehicle\results\model_autoencoder.pth", map_location=device)

# Remove decoder weights if present
filtered_state = {k: v for k, v in state_dict.items() if "encoder" in k}
autoencoder.load_state_dict(filtered_state, strict=False)
autoencoder.eval()
print("✅ Encoder loaded on", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

# ---------------- CARLA Setup ----------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    print("🚗 Vehicle + camera spawned successfully")

    # ---------------- Decision Logic ----------------
    def decide_controls(latent):
        """
        Simple rule-based control using latent mean values.
        Later replace with trained policy network.
        """
        latent_mean = float(latent.mean().cpu())
        steering = np.clip((latent_mean - 0.5) * 2.0, -1.0, 1.0)  # smooth steering
        throttle = 0.6 + 0.2 * random.random()                    # small random accel
        brake = 0.0 if abs(steering) < 0.6 else 0.2
        return steering, throttle, brake, latent_mean

    # ---------------- Process Each Frame ----------------
    def process_img(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        # Encode frame
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = autoencoder(input_tensor)

        steering, throttle, brake, mean_val = decide_controls(latent)
        control = carla.VehicleControl(throttle=throttle, steer=steering, brake=brake)
        vehicle.apply_control(control)

        text = f"Mean:{mean_val:.4f} | Steer:{steering:.2f} | Throttle:{throttle:.2f} | Brake:{brake:.2f}"
        frame_disp = cv2.putText(frame, text, (10, 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Human-Like Control Feed", frame_disp)
        cv2.waitKey(1)

    camera.listen(lambda image: process_img(image))
    print("✅ Driving with human-like encoder control... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🧹 Cleaning up...")
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("✅ Stopped simulation")

if __name__ == "__main__":
    main()
