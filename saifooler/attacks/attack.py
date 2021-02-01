import abc
from typing import Any, Union

import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import pytorch3d.io as py3dio
from torch.nn import CrossEntropyLoss

from saifooler.utils import greyscale_heatmap
from saifooler.viewers.viewer import Viewer3D


class SaifoolerAttack(pl.LightningModule, metaclass=abc.ABCMeta):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh: Union[str, Meshes], render_module, classifier, epsilon,
                 mesh_name="", saliency_maps=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(mesh, Meshes):
            self.mesh: Meshes = mesh.to(self.device)
        elif isinstance(mesh, str):
            self.mesh: Meshes = py3dio.load_objs_as_meshes([mesh], device=self.device)

        self.render_module = render_module
        self.mesh_name = mesh_name

        self.classifier = classifier
        self.epsilon = epsilon

        self.src_texture = self.mesh.textures.maps_padded()

        self.delta = torch.nn.Parameter(
            torch.zeros_like(self.src_texture)
        )
        self.register_parameter("delta", self.delta)

        self.saliency_maps = saliency_maps

        # call update_textures to set the parameter as texture
        # or else it will not use it at first epoch
        self.update_textures()

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.accuracies = {}

    def parameters(self, recurse: bool = True):
        return iter([self.delta])

    def update_textures(self):
        new_maps = self.src_texture + self.delta
        self.mesh.textures.set_maps(new_maps.clamp_(0., 1.))

    def apply_input(self, render_input):
        """

        :param render_input: 1x5 tensor in the form (distance, camera_azim, camera_elev, lights_azim, lights_elev)
        :return: None
        """

        distance, camera_azim, camera_elev = render_input[:3]
        self.render_module.look_at_mesh(distance, camera_azim, camera_elev)

        lights_azim, lights_elev = render_input[3:]
        self.render_module.set_lights_direction(lights_azim, lights_elev)


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
            self.apply_input(render_input)
            image = self.render()
            images.append(image)

        return torch.cat(images, 0)

    def __log_accuracy(self, accuracy, phase: str):
        correct = accuracy.correct.item()
        total = accuracy.total.item()
        acc = accuracy.compute()
        self.accuracies[f"{phase}_accuracy"] = acc
        self.log(f"{self.mesh_name}/{phase}_accuracy", acc, prog_bar=True)
        self.print(f'{phase.capitalize()} accuracy: {correct}/{total} = {acc}')

        if self.current_epoch == 0 and phase == 'train':
            # save the accuracy before attack
            self.accuracies["before_attack"] = acc

    def on_train_epoch_start(self):
        self.train_accuracy.reset()

    def training_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx)
        self.train_accuracy(predictions, targets)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.update_textures()
        for tex_name, tex in self.get_textures().items():
            self.logger.experiment.add_image(f"{self.mesh_name}/textures/{tex_name}", tex.permute(2, 0, 1),
                                             global_step=self.current_epoch)

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
        self.src_texture = self.src_texture.to(device)
        self.render_module.to(device)
        self.classifier.to(device)
        super().to(device)

    def cuda(self, deviceId=None):
        self.to(deviceId)
        super().cuda(deviceId)

    def cpu(self):
        self.to('cpu')
        super().cpu()

    def compute_saliency_maps(self, images):
        images = images.detach().clone()
        images.requires_grad_(True)

        scores = self.classifier.classify(images)
        max_score, _ = scores.max(1, keepdim=True)
        max_score.backward(torch.ones_like(max_score))
        saliency_maps, _ = torch.max(images.grad.data.abs(), dim=3)
        return saliency_maps

    def handle_batch(self, batch, batch_idx):
        render_inputs, targets = batch
        images = self.render_batch(render_inputs)

        if self.saliency_maps and not self.trainer.testing:
            saliency_maps = self.compute_saliency_maps(images)
            heatmaps = greyscale_heatmap(saliency_maps)
            saliency_grid = Viewer3D.make_grid(heatmaps.unsqueeze(3))
            self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{batch_idx}_saliency",
                                             saliency_grid.permute((2, 0, 1)), global_step=self.current_epoch)


        # classify images and extract class predictions
        class_tensors = self.classifier.classify(images)
        _, classes_predicted = class_tensors.max(1, keepdim=True)

        # mask images on which the prediction was wrong
        loss_targets = targets.clone()
        loss_targets[classes_predicted != targets] = -1
        loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=-1)

        # compute CrossEntropyLoss
        loss = loss_fn(class_tensors, loss_targets.squeeze(1))
        images_grid = Viewer3D.make_grid(images)
        self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{batch_idx}",
                                         images_grid.permute((2, 0, 1)), global_step=self.current_epoch)

        return loss, classes_predicted, targets

    @abc.abstractmethod
    def configure_optimizers(self):
        pass