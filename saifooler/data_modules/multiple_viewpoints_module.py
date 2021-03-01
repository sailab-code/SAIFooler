import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader

class MultipleViewModule(pl.LightningDataModule):
    def __init__(self, target_class,
                 distance=0.,
                 orientation_azim_range=(0., 360.),
                 orientation_azim_steps=36,
                 orientation_elev_range=(-90., 90.),
                 orientation_elev_steps=18,
                 light_azim_range=(0., 360.),
                 light_azim_steps=8,
                 light_elev_range=(-90., 90.),
                 light_elev_steps=4,
                 batch_size=None):
        super().__init__()

        self.light_azim_range = light_azim_range
        self.light_azim_steps = light_azim_steps
        self.light_elev_range = light_elev_range
        self.light_elev_steps = light_elev_steps

        self.orientation_azim_range = orientation_azim_range
        self.orientation_azim_steps = orientation_azim_steps
        self.orientation_elev_range = orientation_elev_range
        self.orientation_elev_steps = orientation_elev_steps
        
        self.target_class = target_class
        self.distance = distance

        self.batch_size = self.total_steps if batch_size is None else batch_size
        self.inputs = None
        self.targets = None

    @property
    def total_steps(self):
        return self.light_elev_steps * self.light_azim_steps * self.orientation_azim_steps * self.orientation_elev_steps

    @property
    def number_of_batches(self):
        return self.total_steps // self.batch_size

    def setup(self, stage=None):
        orientations_azim = torch.linspace(*self.orientation_azim_range, self.orientation_azim_steps)
        orientations_elev = torch.linspace(*self.orientation_elev_range, self.orientation_elev_steps)
        lights_azim = torch.linspace(*self.light_azim_range, self.light_azim_steps)
        lights_elev = torch.linspace(*self.light_elev_range, self.light_elev_steps)
        distance = torch.tensor([self.distance])

        inputs = torch.cartesian_prod(distance, orientations_azim, orientations_elev, lights_azim, lights_elev)

        targets = torch.full((inputs.shape[0], 1), self.target_class, dtype=torch.long)
        self.inputs, self.targets = inputs, targets

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.inputs, self.targets), batch_size=self.batch_size)
