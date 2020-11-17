import torch
import pytorch_lightning as pl
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
        targets = torch.tensor([self.target_color])
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)


if __name__ == '__main__':
    tex_module = TextureModule(mesh_path)
    dm = ColorTargetDataModule(torch.tensor([0., 1., 0.]))

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(tex_module, dm.train_dataloader(), None)
