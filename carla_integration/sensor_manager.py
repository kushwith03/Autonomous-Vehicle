import carla
import numpy as np
import cv2
import queue

class SensorManager:
    def __init__(self, world, vehicle, config):
        self.world = world
        self.vehicle = vehicle
        self.config = config['carla']['camera']
        self.image_queue = queue.Queue()
        self.camera = self._setup_camera()

    def _setup_camera(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.config['width']))
        bp.set_attribute('image_size_y', str(self.config['height']))
        bp.set_attribute('fov', str(self.config['fov']))

        loc = self.config['location']
        rot = self.config['rotation']
        transform = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                   carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
        
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        camera.listen(lambda data: self.image_queue.put(data))
        print("Camera sensor initialized.")
        return camera

    def get_latest_image(self):
        try:
            return self.image_queue.get(timeout=2.0)
        except queue.Empty:
            return None

    @staticmethod
    def process_image(carla_image):
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        array = array[:, :, :3]  # RGBA to RGB
        return array
