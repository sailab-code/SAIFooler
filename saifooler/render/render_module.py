from typing import Any, Dict

import pytorch_lightning as pl
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, DirectionalLights, \
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
from pytorch3d.structures import Meshes
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class RenderModule(pl.LightningModule):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, render_settings: Dict = None, *args, **kwargs):
        """

        :param mesh: ``Union[str, Meshes]``: a Meshes object or a string path to the .obj file
        :param render_settings: A dict with optional settings: ``dict: { 'rasterizer': { 'image_size':int, 'blur_radius': float },
            'lights': { 'type': 'point' | 'directional', 'location': list(float,float,float) }``
        """

        super().__init__()
        if render_settings is None:
            render_settings = {}

        R, T = look_at_view_transform(1., 0., 0.)
        self.cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60.0,
            znear=0.01,
            zfar=1000.,
            device=self.device
        )

        raster_settings_dict = {
            'image_size': 224,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': True,
            ** render_settings.get('rasterizer', {})
        }

        raster_settings = RasterizationSettings(**raster_settings_dict)

        lights_settings = {
            'type': 'directional',
            'ambient_color': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), ),
            'diffuse_color': ((0.3, 0.3, 0.3), (0.3, 0.3, 0.3), ),
            'specular_color': ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), ),
            'direction': ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), ),
            ** render_settings.get('lights', {})
        }
        lights_type = lights_settings['type']
        if lights_type == 'point':
            clz = PointLights
        elif lights_type == 'directional':
            clz = DirectionalLights
        else:
            raise ValueError("lights.type can be only 'point' or 'directional'")

        del lights_settings['type']
        lights = clz(**lights_settings, device=self.device)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=lights,
                device=self.device
            )
        )

    def update_camera(self, r, t):
        self.cameras.R, self.cameras.T = r.to(self.device), t.to(self.device)

    def look_at_mesh(self, distance, elevation, azimuth):
        self.update_camera(*look_at_view_transform(distance, elevation, azimuth))

    def render(self, mesh):
        if not isinstance(mesh, Meshes):
            raise ValueError("mesh must be of type Meshes")

        # replicate the mesh so it is as big as the camera inputs
        N = self.cameras.R.shape[0]
        mesh_ext = mesh.extend(N)
        mesh_ext.to(self.device)
        mesh_ext.textures.to(self.device)
        return self.renderer(mesh_ext)[..., :3]

    def forward(self, mesh):
        return self.render(mesh)

    def to(self, device):
        self.renderer.to(device)
        self.cameras.to(device)
        super().to(device)

    def cuda(self, device=None):
        super().cuda(device)
        self.to(device)

    def cpu(self):
        super().cpu()
        self.to('cpu')