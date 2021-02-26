from typing import Any

from pytorch3d.renderer import camera_position_from_spherical_angles, DirectionalLights
from sailenv.agent import Agent
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.viewers.viewer import Viewer3D


class SailenvModule(pl.LightningModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, agent: Agent,
                 lights: DirectionalLights,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent: Agent = agent
        self.set_unity_lights(lights)
        self.obj_id = None

    def set_unity_lights(self, lights):
        main_r, main_g, main_b = (lights.diffuse_color[0] * 255).to(dtype=torch.uint8)
        rim_r, rim_g, rim_b = (lights.ambient_color[0] * 255).to(dtype=torch.uint8)

        self.agent.set_light_color("Main Light", main_r, main_g, main_b)
        self.agent.set_light_color("Rim Light", rim_r, rim_g, rim_b)

    def spawn_obj(self, mesh_descriptor):
        obj_zip = mesh_descriptor.save_to_zip()
        remote_zip = self.agent.send_obj_zip(obj_zip)
        self.obj_id = self.agent.spawn_object(f"file:{remote_zip}")

    def despawn_obj(self):
        self.agent.despawn_object(self.obj_id)

    def look_at_mesh(self, distance, azimuth, elevation):
        position = camera_position_from_spherical_angles(distance, elevation, azimuth)
        rotation = (180-elevation, azimuth, 180)

        self.agent.set_position(list(position.squeeze()))
        self.agent.set_rotation(rotation)

    def set_lights_direction(self, azimuth, elevation):
        # to compute the direction, we can use pytorch3d camera_position_from_spherical_angles.
        # It will return a point in the unit sphere that represent the opposite of our direction
        # distance is fixed at 1 so we get a unit vector

        direction = -camera_position_from_spherical_angles(1., elevation, azimuth)[0]
        direction = direction.detach().cpu().numpy().tolist()
        self.agent.set_light_direction("Main Light", direction)
        self.agent.set_light_direction("Rim Light", direction)

    def render(self, *_):
        frame = torch.tensor(self.agent.get_frame()["main"])

        return torch.fliplr(frame).clone().unsqueeze(0).to(self.device)

    def to(self, device):
        super().to(device)

    def cuda(self, device=None):
        super().cuda(device)
        self.to(device)

    def cpu(self):
        super().cpu()
        self.to('cpu')