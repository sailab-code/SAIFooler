from typing import Union, List

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class OrientationDataModule(pl.LightningDataModule):
    def __init__(self, target_class, elev=0., distance=0., steps=6, batch_size=None):
        super().__init__()
        self.target_class = target_class
        self.elev = elev
        self.distance = distance
        self.steps = steps
        self.batch_size = batch_size if batch_size is not None else steps
        self.inputs = None
        self.targets = None

    def setup(self, stage=None):
        orientations = torch.linspace(0., 360., self.steps)

        inputs = torch.zeros((orientations.shape[0], 3))
        inputs[:, 0], inputs[:, 1], inputs[:, 2] = self.distance, self.elev, orientations

        targets = torch.full_like(orientations, self.target_class, dtype=torch.long)
        self.inputs, self.targets = inputs, targets

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)
