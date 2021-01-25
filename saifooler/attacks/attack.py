import abc
from typing import Any, Union

import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import pytorch3d.io as py3dio

class SaifoolerAttack(pl.LightningModule, metaclass=abc.ABCMeta):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh: Union[str, Meshes], render_module, classifier, epsilon, mesh_name="", mini_batch_size=6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(mesh, Meshes):
            self.mesh: Meshes = mesh.to(self.device)
        elif isinstance(mesh, str):
            self.mesh: Meshes = py3dio.load_objs_as_meshes([mesh], device=self.device)

        self.render_module = render_module
        self.mesh_name = mesh_name

        self.mini_batch_size = mini_batch_size

        self.classifier = classifier
        #for p in self.classifier.parameters():
        #    p.requires_grad_(False)
        self.epsilon = epsilon

        self.texture = torch.nn.Parameter(
            self.mesh.textures.maps_padded()
        )
        self.register_parameter("texture", self.texture)
        # call update textures to set the parameter as texture
        # or else it will not use it at first epoch
        self.update_textures()

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.accuracies = {}

    def parameters(self, recurse: bool = True):
        return iter([self.texture])

    def update_textures(self):
        self.mesh.textures.set_maps(self.texture)

    def apply_input(self, distance, elevation, azimuth):
        self.render_module.look_at_mesh(distance, elevation, azimuth)

    def render(self):
        return self.render_module.render(self.mesh)

    def get_textures(self):
        textures = self.mesh.textures.get_textures()
        return {
            tex_name: texture.cpu()
            for tex_name, texture in textures.items()
        }

    def render_batch(self, render_inputs):
        # we must render each image separately, batch rendering is broken for some reason
        images = []
        for render_input in render_inputs:
            self.apply_input(*render_input)
            image = self.render()
            images.append(image)

        return torch.cat(images, 0)

    def __log_accuracy(self, accuracy, phase: str):
        correct = accuracy.correct.item()
        total = accuracy.total.item()
        acc = accuracy.compute()
        self.accuracies[f"{phase}_accuracy"] = acc
        self.log(f"{phase}_accuracy", acc)
        self.print(f'{phase.capitalize()} accuracy: {correct}/{total} = {acc}')

    def on_train_epoch_start(self):
        self.train_accuracy.reset()

    def training_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx)
        self.train_accuracy(predictions, targets)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.update_textures()

    def on_train_epoch_end(self, outputs):
        self.__log_accuracy(self.train_accuracy, "train")

    def on_validation_epoch_start(self) -> None:
        self.valid_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx)
        self.valid_accuracy(predictions, targets)
        return total_loss

    def on_validation_epoch_end(self):
        self.__log_accuracy(self.valid_accuracy, "validation")

    def on_test_epoch_start(self) -> None:
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx)
        self.test_accuracy(predictions, targets)
        return total_loss

    def on_test_epoch_end(self):
        self.__log_accuracy(self.test_accuracy, "test")

    def to(self, device):
        self.mesh = self.mesh.to(device)
        self.mesh.textures = self.mesh.textures.to(device)
        self.render_module.to(device)
        self.classifier.to(device)
        super().to(device)

    def cuda(self, deviceId=None):
        self.to(deviceId)
        super().cuda(deviceId)

    def cpu(self):
        self.to('cpu')
        super().cpu()

    def handle_batch(self, batch, batch_idx):
        render_inputs, targets = batch

        batch_losses, batch_predictions = [], []
        # split batch into several mini batches
        for i in range(0, render_inputs.shape[0], self.mini_batch_size):
            mini_batch_slice = slice(i, i+self.mini_batch_size)
            mini_batch_inputs = render_inputs[mini_batch_slice]
            mini_batch_targets = targets[mini_batch_slice]
            mini_batch_idx = int(i / self.mini_batch_size)
            mini_batch_output = self.handle_mini_batch((mini_batch_inputs, mini_batch_targets), mini_batch_idx)
            loss, predictions = mini_batch_output
            batch_losses.append(loss)
            batch_predictions.append(predictions)

        total_loss = sum(batch_losses)
        predictions = torch.cat(batch_predictions, 0)

        return total_loss, predictions, targets

    @abc.abstractmethod
    def handle_mini_batch(self, mini_batch, mini_batch_idx):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass