import torch
import pytorch_lightning as pl
from pytorch3d.renderer import look_at_view_transform
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler.texture_atlas_module import TextureAtlasModule
from saifooler.texture_module import TextureModule
import matplotlib.pyplot as plt

mesh_path = "./meshes/toilet/toilet.obj"


class ColorTargetDataModule(pl.LightningDataModule):
    def __init__(self, target_color):
        super().__init__()
        self.target_color = target_color

    def train_dataloader(self):
        inps = torch.tensor([[3.5, 60., 125.]])
        targets = self.target_color.unsqueeze(0)
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)


if __name__ == '__main__':
    tex_module = TextureAtlasModule(mesh_path, texture_atlas_size=4)
    dm = ColorTargetDataModule(torch.tensor([1., 0., 0.]))

    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(tex_module, dm.train_dataloader(), None)

    tex_module.show_textures()

    viewpoints = [
        (3.5, 90., 90.),
        (3.5, 45., 180.),
        (3.5, 30., 124.),
        (3.5, 45., 275.),
        (3.5, -90., 90.),
        (3.5, -45., 180.),
        (3.5, -30., 124.),
        (3.5, -45., 275.)
    ]

    for vp in viewpoints:
        tex_module.show_render(vp)



