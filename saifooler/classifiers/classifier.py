from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class Classifier(pl.LightningModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, model, *args, **kwargs):
        super().__init__()

        self.model = model
        self.model.eval()

    def normalize_image(self, image):
        """
        Normalizes the image for use of a particular classifier.
        Extend this method if you need normalization
        :param image: image to be normalized
        :return: normalized image
        """

        # a generic classifier will not normalize.
        return image

    def parameters(self, recurse: bool = True):
        return iter([])

    def classify(self, image):
        self.model.eval()
        normalized_image = self.normalize_image(image)
        return self.model(normalized_image)

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.model.eval()

    def cuda(self, device=None):
        super().cuda(device)
        self.to(device)

    def cpu(self):
        super().cpu()
        self.to('cpu')

