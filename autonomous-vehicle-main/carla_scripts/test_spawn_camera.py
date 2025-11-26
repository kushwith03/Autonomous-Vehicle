import carla
import random
import time
import numpy as np
import cv2

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # --- Attach RGB camera ---
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    def process_img(image):
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        cv2.imshow("CARLA Camera", img)
        cv2.waitKey(1)

    camera.listen(lambda image: process_img(image))
    print("✅ Vehicle spawned & camera streaming. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Destroying actors...")
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
