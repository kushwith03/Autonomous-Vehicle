# File: E:\AutonomousVehicle\integrate_model.py
# Integrates your trained AutoEncoder model with CARLA simulator for live perception

import sys
import os
import time
import torch
import numpy as np
import cv2
from torchvision import transforms

# =====================================================
# ✅  Add CARLA Python API paths manually
# =====================================================
CARLA_EGG_PATH = r"E:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg"
CARLA_PY_PATH  = r"E:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla"

for p in [CARLA_EGG_PATH, CARLA_PY_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

import carla  # import after paths are added
from train_test import AutoEncoder  # reuse your trained model architecture

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"E:\AutonomousVehicle\results\model_autoencoder.pth"
IMG_SIZE = (800, 600)
device = torch.device("cpu")

# =====================================================
# LOAD MODEL
# =====================================================
model = AutoEncoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Loaded model from:", MODEL_PATH)

# Transform
to_tensor = transforms.ToTensor()

def carla_to_tensor(image):
    """Convert CARLA raw image to tensor."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, IMG_SIZE)
    tensor = to_tensor(frame_resized).unsqueeze(0).to(device)
    return frame_resized, tensor

def show_reconstruction(frame, tensor):
    """Run model on frame tensor and show original vs reconstruction."""
    with torch.no_grad():
        recon = model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon = np.clip(recon * 255.0, 0, 255).astype(np.uint8)
    combined = np.hstack((frame, recon))
    cv2.imshow("Original (Left) | Reconstructed (Right)", combined)
    cv2.waitKey(1)

# =====================================================
# MAIN FUNCTION
# =====================================================
def main():
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("✅ Connected to world:", world.get_map().name)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("🚗 Vehicle spawned:", vehicle.type_id)

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_SIZE[0]))
    cam_bp.set_attribute('image_size_y', str(IMG_SIZE[1]))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print("📸 Camera attached to vehicle.")

    def process_img(image):
        frame, tensor = carla_to_tensor(image)
        show_reconstruction(frame, tensor)

    camera.listen(process_img)
    print("🎥 Streaming live frames... (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("✅ Cleaned up actors and closed windows.")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()
