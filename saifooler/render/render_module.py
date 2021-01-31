from typing import Any, Dict

import pytorch_lightning as pl
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, DirectionalLights, \
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, camera_position_from_spherical_angles
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

        lights_settings = render_settings.get('lights', {})
        lights_type = lights_settings.get("type", "directional")

        if lights_type == 'point':
            lights_settings = {
                "type": "point",
                "location": [[0.5, 5.0, 0.0]],
                **lights_settings
            }
            clz = PointLights
        elif lights_type == 'directional':
            lights_settings = {
                "type": "directional",
                "direction": [[0., 1., 0.]],
                **lights_settings
            }
            clz = DirectionalLights
        else:
            raise ValueError("lights.type can be only 'point' or 'directional'")

        del lights_settings['type']
        self.lights = clz(**lights_settings, device=self.device)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=self.lights,
                device=self.device
            )
        )

    def update_camera(self, r, t):
        self.cameras.R, self.cameras.T = r.to(self.device), t.to(self.device)

    def look_at_mesh(self, distance, azimuth, elevation):
        self.update_camera(*look_at_view_transform(distance, elevation, azimuth))

    def set_lights_direction(self, azimuth, elevation):
        if isinstance(self.lights, DirectionalLights):
            # to compute the direction, we can use pytorch3d camera_position_from_spherical_angles.
            # It will return a point in the unit sphere that represent the opposite of our direction
            # distance is fixed at 1 so we get a unit vector

            direction = -camera_position_from_spherical_angles(1., elevation, azimuth)
            self.lights.direction = direction

    def render(self, mesh):
        if not isinstance(mesh, Meshes):
            raise ValueError("mesh must be of type Meshes")

        # replicate the mesh so it is as big as the camera inputs
        N = self.cameras.R.shape[0]

        if N > 1:
            mesh_ext = mesh.extend(N)
            mesh_ext.to(self.device)
            mesh_ext.textures.to(self.device)
        else:
            mesh_ext = mesh

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