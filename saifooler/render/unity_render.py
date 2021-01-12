from pytorch3d.renderer import look_at_rotation, camera_position_from_spherical_angles
from sailenv.agent import Agent
import numpy as np

class UnityRender:
    def __init__(self, agent: Agent, obj_zip):
        self.agent = agent
        self.obj_zip = obj_zip

    def change_scene(self):
        self.agent.change_scene("object_view/scene")

    def spawn_obj(self):
        remote_zip = self.agent.send_obj_zip(self.obj_zip)
        self.agent.spawn_object(f"file:{remote_zip}")

    def look_at_mesh(self, distance, elevation, azimuth):
        position = camera_position_from_spherical_angles(distance, elevation, azimuth)
        rotation = (-elevation, 180+azimuth, 0)

        self.agent.set_position(list(position.squeeze()))
        self.agent.set_rotation(rotation)

    def render(self):
        frame = self.agent.get_frame()
        return np.fliplr(frame["main"])