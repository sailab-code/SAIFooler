from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler import default_renderer, default_device
from saifooler.render_module import RenderModule

import matplotlib.pyplot as plt


class MeshModule(RenderModule):
    def __init__(self, mesh_path, renderer=default_renderer, device=default_device, *args, **kwargs):
        super().__init__(mesh_path, renderer, device, *args, **kwargs)
        verts_shape = self.mesh.verts_packed().shape
        self.deform_verts = torch.full(verts_shape, 0., device=self.device, requires_grad=True)

    def apply_deform(self):
        return self.mesh.offset_verts(self.deform_verts)

    def render(self):
        new_mesh = self.apply_deform()
        return self.renderer(new_mesh)[0, ..., :3]

    def forward(self):
        return self.render()

    def training_step(self, batch, batch_idx):
        image = self.render()
        _, target_img = batch
        loss = torch.mean(torch.pow(image - target_img.cuda(), 2))
        return loss

    def validation_step(self, batch, batch_idx):
        image = self.render()
        _, target_img = batch
        loss = torch.mean(torch.pow(image - target_img.cuda(), 2))
        return loss

    def configure_optimizers(self):
        return Adam([self.deform_verts], lr=0.001)

    def show_render(self):
        image = self.render().cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.grid("off")
        plt.axis("off")
        plt.show()

    def on_train_start(self) -> None:
        self.show_render()

    def on_validation_model_train(self) -> None:
        self.show_render()

    def on_train_end(self) -> None:
        self.show_render()