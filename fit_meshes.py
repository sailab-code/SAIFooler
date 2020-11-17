import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler.mesh_module import MeshModule
from saifooler.render_module import RenderModule
from saifooler.texture_module import TextureModule
import matplotlib.pyplot as plt

book_mesh_path = "./meshes/book/book_open_01.obj"
candle_mesh_path = "./meshes/candle/candle_test.obj"


class TargetImageDataModule(pl.LightningDataModule):
    def __init__(self, target_image):
        super().__init__()
        self.target_image = target_image

    def train_dataloader(self):
        inps = torch.tensor([[0.]])
        targets = self.target_image.unsqueeze(0)
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)


if __name__ == '__main__':
    book_module = MeshModule(book_mesh_path)
    candle_module = RenderModule(candle_mesh_path)
    candle_image = candle_module.render()
    dm = TargetImageDataModule(candle_image)

    trainer = pl.Trainer(max_epochs=500, check_val_every_n_epoch=5)
    trainer.fit(book_module, dm.train_dataloader(), dm.train_dataloader())