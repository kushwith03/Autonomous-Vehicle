"""
record_latent_data.py
----------------------------------------
Records CARLA autopilot driving data and corresponding AutoEncoder latent vectors.
Works safely in synchronous mode, ticking the world manually (no freeze / crash).
Auto-saves every 100 samples to E:\AutonomousVehicle\results\latent_drive_data_YYYYMMDD_HHMMSS.csv
"""

import carla
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import time
from torchvision import transforms
from datetime import datetime

# ============================================================
#                AutoEncoder Definition
# ============================================================
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


# ============================================================
#               Load Trained AutoEncoder
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = AutoEncoder().to(device)
state_dict = torch.load(r"E:\AutonomousVehicle\results\model_autoencoder.pth", map_location=device)

# Load only encoder weights
filtered = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder" in k}
encoder.encoder.load_state_dict(filtered, strict=False)
encoder.eval()

print("✅ Encoder loaded on", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

# ============================================================
#                   Recording Logic
# ============================================================
def main():
    # --- Connect to CARLA ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # --- Activate synchronous mode ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.066  # ~15 Hz physics tick
    world.apply_settings(settings)
    print("✅ Synchronous autopilot mode activated.")

    # --- Blueprints ---
    bp = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = bp.filter('model3')[0]
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn)
    vehicle.set_autopilot(True)
    print("🚗 Vehicle spawned and autopilot enabled.")

    # Spawn RGB camera
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print("📸 Camera attached.")

    # Data containers
    data_records = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"E:/AutonomousVehicle/results/latent_drive_data_{timestamp}.csv"

    # ---------------- Helper: Save partial data ----------------
    def save_partial():
        if len(data_records) == 0:
            return
        latent_size = len(data_records[0]) - 3
        cols = ["throttle", "steer", "brake"] + [f"latent_{i}" for i in range(latent_size)]
        df = pd.DataFrame(data_records, columns=cols)
        df.to_csv(out_file, index=False, float_format="%.5f")
        print(f"💾 Auto-saved {len(data_records)} samples → {out_file}")

    # ---------------- Camera callback ----------------
    def process_img(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        # Encode frame
        tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = encoder(tensor).cpu().numpy().flatten()

        # Read autopilot control
        ctrl = vehicle.get_control()
        rec = np.concatenate(([ctrl.throttle, ctrl.steer, ctrl.brake], latent))
        data_records.append(rec)

        # Display & progress
        if len(data_records) % 50 == 0:
            print(f"Samples recorded: {len(data_records)}")
        if len(data_records) % 100 == 0:
            save_partial()

        cv2.imshow("Latent Data Recorder", frame)
        cv2.waitKey(1)

    # Start camera stream
    camera.listen(lambda img: process_img(img))
    print("✅ Recording started — press Ctrl+C once to stop safely")

    # --- Main synchronous loop (ticks world manually) ---
    try:
        while True:
            world.tick()  # <---- advances simulation
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🧹 Stopping and saving final data...")
    finally:
        # Stop safely
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        save_partial()
        print(f"✅ Final data saved at: {out_file}")

        # Reset to async mode (important)
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("🔄 Restored async mode — done.")


# ============================================================
#                     Run Script
# ============================================================
if __name__ == "__main__":
    main()
