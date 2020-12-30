from typing import Any

import torch
from pytorch3d.renderer import look_at_view_transform
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler.render_module import RenderModule
import matplotlib.pyplot as plt


class TextureModule(RenderModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh_path, renderer=None, *args, **kwargs):
        super().__init__(mesh_path, renderer, *args, **kwargs)
        self.tex_filter = torch.nn.Parameter(torch.full(self.mesh.textures.maps_padded().shape, 0.0, device=self.device))

    def apply_filter(self):
        new_textures = self.mesh.textures.clone()
        maps = new_textures.maps_padded()
        maps = maps + self.tex_filter
        maps = maps.clamp(0., 1.)
        new_textures.set_maps(maps)
        return new_textures

    def render(self, camera_params=None):
        if camera_params is not None:
            self.change_camera(*look_at_view_transform(*camera_params))

        new_mesh = self.mesh.clone()
        new_mesh.textures = self.apply_filter()
        return self.renderer(new_mesh)[0, ..., :3]

    def training_step(self, batch, batch_idx):
        camera_params, target_color = batch
        image = self.render(camera_params.squeeze(0))

        loss = torch.mean(torch.pow(image-target_color, 2))
        return loss

    def configure_optimizers(self):
        return Adam([self.tex_filter], lr=0.5)

    def show_textures(self, title=""):
        textures = self.mesh.textures.get_textures()
        for texture in textures:
            plt.figure(figsize=(7, 7))
            #texture_image = self.apply_filter().maps_padded()
            texture_image = texture
            plt.imshow(texture_image.detach().cpu().numpy())
            plt.title(title)
            plt.grid("off")
            plt.axis("off")
            plt.show()

    def on_train_start(self) -> None:
        self.show_render()
        self.show_textures()

    def on_train_end(self) -> None:
        self.show_render()
        self.show_textures()
