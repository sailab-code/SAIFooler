import torch
import pytorch_lightning as pl
from pytorch3d.renderer import look_at_view_transform
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler.texture_module import TextureModule
import matplotlib.pyplot as plt

mesh_path = "./meshes/book/book_open_01.obj"


class ColorTargetDataModule(pl.LightningDataModule):
    def __init__(self, target_color):
        super().__init__()
        self.target_color = target_color

    def train_dataloader(self):
        inps = torch.tensor([[0.]])
        targets = self.target_color.unsqueeze(0)
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)


if __name__ == '__main__':
    tex_module = TextureModule(mesh_path)
    dm = ColorTargetDataModule(torch.tensor([0., 1., 0.]))

    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(tex_module, dm.train_dataloader(), None)

    R, T = look_at_view_transform(0.5, 90., 0.)
    tex_module.change_camera(R, T)
    tex_module.show_render()
