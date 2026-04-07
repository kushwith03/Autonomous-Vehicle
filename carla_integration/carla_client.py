import carla

class CarlaClient:
    def __init__(self, host='localhost', port=2000, timeout=10.0):
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.map = self.world.get_map()
            self.actors = []
            print(f"Connected to CARLA at {host}:{port}")
        except Exception as e:
            print(f"Error: Failed to connect to CARLA: {e}")
            raise

    def set_weather(self, weather_str):
        weather = getattr(carla.WeatherParameters, weather_str)
        self.world.set_weather(weather)

    def spawn_vehicle(self, model='vehicle.audi.a2'):
        bp = self.blueprint_library.filter(model)[0]
        spawn_point = self.map.get_spawn_points()[0]
        vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actors.append(vehicle)
        print(f"Vehicle spawned: {model} at {spawn_point.location}")
        return vehicle

    def cleanup(self):
        for actor in self.actors:
            if actor is not None:
                actor.destroy()
        self.actors = []
        print("Cleanup: CARLA actors destroyed.")
