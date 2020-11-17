import torch
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

import saifooler
from saifooler import default_renderer, default_device
import pytorch_lightning as pl
import pytorch3d
import pytorch3d.io as py3dio
from pytorch3d.renderer import TexturesUV

class RenderModule(pl.LightningModule):
    def _forward_unimplemented(self, *input) -> None:
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
