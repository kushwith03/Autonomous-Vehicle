# Phase 5 — Full Autonomous Driving Integration
# Combines AutoEncoder (vision) + Controller (decision) to drive in CARLA

import carla
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from torchvision import transforms

# ---------------- AutoEncoder (encoder only) ----------------
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

# ---------------- Decision Controller ----------------
class DecisionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DecisionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()  # steer, throttle, brake in [-1,1]
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Load Models ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder
encoder = AutoEncoder().to(device)
ae_state = torch.load(r"E:\AutonomousVehicle\results\model_autoencoder.pth", map_location=device)
filtered = {k: v for k, v in ae_state.items() if "encoder" in k}
encoder.load_state_dict(filtered, strict=False)
encoder.eval()

# Load controller
latent_size = 128 * 8 * 8  # same as encoder output size
controller = DecisionModel(latent_size).to(device)
controller.load_state_dict(torch.load(r"E:\AutonomousVehicle\results\controller_model.pth", map_location=device))
controller.eval()

print("✅ Models loaded successfully on", device)

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

    print("🚗 Vehicle + camera ready for autonomous driving")

    def process_img(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = encoder(input_tensor).flatten(1)
            pred = controller(latent).cpu().numpy()[0]

        steer = float(pred[0]) * 0.5                     # smoother steering
        throttle = max(0.3, abs(float(pred[1])))         # ensure minimum forward motion
        brake = 0.0                                      # disable brake for testing

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.apply_control(control)

        text = f"Steer:{steer:.2f} | Throttle:{throttle:.2f} | Brake:{brake:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow("Autonomous Drive", frame)
        cv2.waitKey(1)

    camera.listen(lambda img: process_img(img))
    print("✅ Autonomous driving started... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🧹 Cleaning up...")
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
