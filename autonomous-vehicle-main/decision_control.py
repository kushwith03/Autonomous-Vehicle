# File: E:\AutonomousVehicle\decision_control.py
# Human-like decision controller for autonomous driving using trained AutoEncoder features.

import sys, os, time
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms

# --- Add CARLA paths ---
CARLA_EGG_PATH = r"E:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg"
CARLA_PY_PATH  = r"E:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla"
for p in [CARLA_EGG_PATH, CARLA_PY_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    import carla
except ImportError:
    print("❌ CARLA not found — check PYTHONPATH!")
    raise

from train_test import AutoEncoder

# --- Config ---
MODEL_PATH = r"E:\AutonomousVehicle\results\model_autoencoder.pth"
device = torch.device("cpu")
IMG_SIZE = (800, 600)

# =========================================================
#  Controller Model (Fixed latent_dim = 8192)
# =========================================================
class ControlNet(nn.Module):
    """Neural Controller: maps encoded latent features → [steer, throttle, brake]."""
    def __init__(self, latent_dim=8192):  # Fixed to match trained model
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()  # outputs in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
#  Load Models
# =========================================================
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
autoencoder.eval()

latent_dim = 8192  # fixed to match controller_model.pth
controller = ControlNet(latent_dim).to(device)
print("✅ AutoEncoder loaded; ControlNet initialized (latent_dim=8192).")

# =========================================================
#  Helper Functions
# =========================================================
to_tensor = transforms.ToTensor()

def preprocess(image):
    """Converts CARLA image → normalized tensor for AutoEncoder."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame, IMG_SIZE)
    tensor = to_tensor(resized).unsqueeze(0).to(device)
    return frame, tensor


def predict_control(tensor):
    """Encodes scene → latent → predicts [steer, throttle, brake]."""
    with torch.no_grad():
        encoded = autoencoder.encoder(tensor)
        z = encoded.view(1, -1)
        if z.shape[1] != 8192:
            z = torch.nn.functional.adaptive_avg_pool1d(z.unsqueeze(0), 8192).squeeze(0)
        action = controller(z).cpu().numpy().flatten()

    steer, throttle, brake = action
    steer = float(np.clip(steer, -1.0, 1.0))
    throttle = float(np.clip((throttle + 1) / 2, 0.0, 1.0))  # scale [-1,1] → [0,1]
    brake = float(np.clip((brake + 1) / 2, 0.0, 1.0))
    return steer, throttle, brake


# =========================================================
#  Main CARLA Integration Loop
# =========================================================
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = bp_lib.filter('vehicle.audi.a2')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("🚗 Vehicle spawned successfully.")

    # Attach camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_SIZE[0]))
    cam_bp.set_attribute('image_size_y', str(IMG_SIZE[1]))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print("📸 Camera attached. Starting live AI control...")

    def process(image):
        frame, tensor = preprocess(image)
        steer, throttle, brake = predict_control(tensor)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.apply_control(control)

        display = frame.copy()
        txt = f"Steer:{steer:.2f}  Throttle:{throttle:.2f}  Brake:{brake:.2f}"
        cv2.putText(display, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("AI Driving (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    camera.listen(process)

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🛑 Stopping AI control...")
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("✅ Clean exit — resources released.")


if __name__ == "__main__":
    main()
