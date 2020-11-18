from typing import Any

import torch
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

    def render(self):
        new_mesh = self.mesh.clone()
        new_mesh.textures = self.apply_filter()
        return self.renderer(new_mesh)[0, ..., :3]

    def forward(self):
        return self.render()

    def training_step(self, batch, batch_idx):
        image = self.render()
        _, target_color = batch

        loss = torch.mean(torch.pow(image-target_color, 2))
        return loss

    def configure_optimizers(self):
        return Adam([self.tex_filter], lr=0.5)

    def show_render(self):
        image = self.render().cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.grid("off")
        plt.axis("off")
        plt.show()

    def show_texture(self):
        plt.figure(figsize=(7, 7))
        texture_image = self.apply_filter().maps_padded()
        plt.imshow(texture_image.squeeze().detach().cpu().numpy())
        plt.grid("off")
        plt.axis("off")
        plt.show()

    def on_train_start(self) -> None:
        self.show_render()
        self.show_texture()

    def on_train_end(self) -> None:
        self.show_render()
        self.show_texture()
