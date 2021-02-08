import abc
from typing import Any, Union

from PIL import Image, ImageEnhance
import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import pytorch3d.io as py3dio
from torch.nn import CrossEntropyLoss

from saifooler.render.render_module import RenderModule
from saifooler.utils import greyscale_heatmap
from saifooler.viewers.viewer import Viewer3D

import torchvision.transforms.functional as TF


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

        self.render_module: RenderModule = render_module
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

    def get_view2tex_map(self):
        return self.render_module.get_view2tex_map(self.mesh)

    def get_textures(self):
        textures = self.mesh.textures.get_textures()
        return {
            tex_name: texture.cpu()
            for tex_name, texture in textures.items()
        }

    def render_batch(self, render_inputs):
        # we must render each image separately, batch rendering is broken for some reason
        images = []
        view2tex_maps = []
        for render_input in render_inputs:
            self.apply_input(render_input)
            image = self.render()
            view2tex_map = self.get_view2tex_map()
            images.append(image)
            view2tex_maps.append(view2tex_map)

        return torch.cat(images, 0), torch.cat(view2tex_maps, 0)

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
        # todo: check if can backward on max_score.sum()
        max_score.backward(torch.ones_like(max_score))
        saliency_maps, _ = torch.max(images.grad.data.abs(), dim=3, keepdim=True)
        return saliency_maps

    def handle_batch(self, batch, batch_idx):
        """
        N batch size, WxH view size
        :param batch:
        :param batch_idx:
        :return:
        """
        render_inputs, targets = batch
        images, view2tex_maps = self.render_batch(render_inputs)
        batch_size = render_inputs.shape[0]

        images_grid = Viewer3D.make_grid(images)
        self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{batch_idx}",
                                         images_grid.permute((2, 0, 1)), global_step=self.current_epoch)

        if self.saliency_maps and not self.trainer.testing:
            saliency_maps = self.compute_saliency_maps(images) # NxWxHx1 between 0..inf
            heatmaps = greyscale_heatmap(saliency_maps) # NxWxHx1 between 0..1
            saliency_grid = Viewer3D.make_grid(heatmaps)
            self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{batch_idx}_view_saliency",
                                             saliency_grid.permute((2, 0, 1)), global_step=self.current_epoch)

            #  shape is (1 x Wt x Ht x 3), the first and last dimensions are omitted, leaving (Wt x Ht)
            tex_shape = self.mesh.textures.maps_padded().shape[1:-1]
            view2tex_maps = (view2tex_maps * torch.tensor(tex_shape, device=self.device)).to(dtype=torch.long)
            tex_saliency = torch.zeros(tex_shape, device=self.device)

            for idx in range(batch_size):
                tex_saliency[(view2tex_maps[idx, ..., 1], view2tex_maps[idx, ..., 0])] += saliency_maps.squeeze(3)[idx]
            # todo: divide for batch_size in optimizer


            heatmaps = greyscale_heatmap(tex_saliency.unsqueeze(0).unsqueeze(3))  # NxWxHx1 between 0..1

            red_heatmap = torch.zeros((*heatmaps.shape[1:-1], 3))
            red_heatmap[..., 0] = heatmaps[..., 0]
            heatmap_img = TF.to_pil_image(red_heatmap.permute(2, 0, 1))
            tex_img = TF.to_pil_image(self.src_texture.squeeze(0).cpu().permute(2, 0, 1))

            blended = Image.blend(heatmap_img, tex_img, 0.1)
            brightness_enhance = ImageEnhance.Brightness(blended)
            blended = brightness_enhance.enhance(2.5)

            blended = TF.pil_to_tensor(blended).to(dtype=torch.float32) / 255
            blended = blended.permute(1, 2, 0).unsqueeze(0)


            saliency_grid = Viewer3D.make_grid(blended)
            self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{batch_idx}_tex_saliency",
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

        return loss, classes_predicted, targets

    @abc.abstractmethod
    def configure_optimizers(self):
        pass
