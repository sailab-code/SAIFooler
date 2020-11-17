from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

import saifooler
import pytorch_lightning as pl
import pytorch3d
import pytorch3d.io as py3dio
from pytorch3d.renderer import TexturesUV

from saifooler import default_renderer, default_device


class TextureModule(pl.LightningModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh_path, renderer=None, device=default_device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if renderer is None:
            self.renderer = default_renderer
        else:
            self.renderer = renderer

        if device is None:
            self.to(default_device)
        else:
            self.to(device)

        self.mesh = py3dio.load_objs_as_meshes([mesh_path], device=self.device)
        self.tex_filter = torch.full(self.mesh.textures.maps_padded().shape, 0.0, device=self.device, requires_grad=True)

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

    def train_dataloader(self):
        inps = torch.tensor([[0.]], device=self.device)
        targets = torch.tensor([[0.,0., 1.]], device=self.device)
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)

    def configure_optimizers(self):
        return Adam([self.tex_filter], lr=0.5)


