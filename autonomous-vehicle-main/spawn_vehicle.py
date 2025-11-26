# File: E:\AutonomousVehicle\spawn_vehicle.py
import carla
import time
import os
import random

RESULTS_DIR = r"E:\AutonomousVehicle\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # choose first vehicle blueprint
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("No spawn points found in the map.")
transform = random.choice(spawn_points)

vehicle = None
camera = None
actor_list = []

try:
    vehicle = world.spawn_actor(vehicle_bp, transform)
    actor_list.append(vehicle)
    print("✅ Spawned vehicle:", vehicle.type_id)

    # Create RGB camera
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')

    # attach camera to vehicle at a front position
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    actor_list.append(camera)
    print("✅ Camera attached.")

    # callback to save images
    def save_image(image):
        filename = os.path.join(RESULTS_DIR, f"image_{image.frame:06d}.png")
        image.save_to_disk(filename)
        print("Saved image:", filename)

    camera.listen(save_image)

    # Let it run for 10 seconds to collect some images
    print("Collecting images for 10 seconds...")
    time.sleep(10)

finally:
    print("Cleaning up actors...")
    for a in actor_list:
        try:
            a.destroy()
        except Exception:
            pass
    print("Done.")
