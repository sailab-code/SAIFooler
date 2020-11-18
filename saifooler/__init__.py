import pytorch3d
import torch

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# extend TexturesUV
def set_maps(self, maps):
    from pytorch3d.renderer.mesh.textures import _pad_texture_maps
    if torch.is_tensor(maps):
        if maps.ndim != 4 or maps.shape[0] != self._N:
            msg = "Expected maps to be of shape (N, H, W, 3); got %r"
            raise ValueError(msg % repr(maps.shape))
        self._maps_padded = maps
        self._maps_list = None
    elif isinstance(maps, (list, tuple)):
        if len(maps) != self._N:
            raise ValueError("Expected one texture map per mesh in the batch.")
        self._maps_list = maps
        if self._N > 0:
            maps = _pad_texture_maps(maps, align_corners=self.align_corners)
        else:
            maps = torch.empty(
                (self._N, 0, 0, 3), dtype=torch.float32, device=self.device
            )
        self._maps_padded = maps
    else:
        raise ValueError("Expected maps to be a tensor or list.")

TexturesUV.set_maps = set_maps
