from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

import saifooler
import pytorch_lightning as pl
import pytorch3d
import pytorch3d.io as py3dio
from pytorch3d.renderer import TexturesUV, MeshRenderer, MeshRasterizer, look_at_view_transform, FoVPerspectiveCameras, \
    RasterizationSettings, SoftPhongShader, PointLights
import matplotlib.pyplot as plt

class RenderModule(pl.LightningModule):
    def _forward_unimplemented(self, *input) -> None:
        pass

    def __init__(self, mesh_path, renderer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = self.__make_default_renderer()

        self.rasterizer = self.renderer.rasterizer
        self.shader = self.renderer.shader
        self.cameras = self.rasterizer.cameras
        self.lights = self.shader.lights

        self.mesh = py3dio.load_objs_as_meshes([mesh_path], device=self.device)

    def cuda(self, deviceId=None):
        super().cuda(deviceId)
        self.to(self.device)

    def cpu(self):
        super().cpu()
        self.to('cpu')

    def to(self, device):
        super().to(device)
        self.mesh = self.mesh.to(device)
        self.renderer.to(device)

    def __make_default_renderer(self):
        R, T = look_at_view_transform(0.5, 10, 180)
        default_camera = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        default_raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        default_lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=default_camera,
                raster_settings=default_raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=default_camera,
                lights=default_lights
            )
        )

    def render(self, camera_params=None):
        if camera_params is not None:
            self.change_camera(*look_at_view_transform(*camera_params))
        return self.renderer(self.mesh)[0, ..., :3]

    def change_camera(self, r, t):
        cameras = self.cameras
        cameras.R, cameras.T = r.to(self.device), t.to(self.device)

    def show_render(self, camera_params=None):
        image = self.render(camera_params).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.grid("off")
        plt.axis("off")
        plt.show()