import abc
from typing import Any, Union

from PIL import Image, ImageEnhance
import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import pytorch3d.io as py3dio
from torch.nn import CrossEntropyLoss

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.render.sailenv_module import SailenvModule
from saifooler.utils import greyscale_heatmap
from saifooler.viewers.viewer import Viewer3D

import torchvision.transforms.functional as TF

PYTORCH3D_MODULE_NAME = "pytorch3d"
SAILENV_MODULE_NAME = "sailenv"

class SaifoolerAttack(pl.LightningModule, metaclass=abc.ABCMeta):


    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh_descriptor: MeshDescriptor, pytorch3d_module, sailenv_module, classifier, epsilon,
                saliency_maps=None, saliency_threshold=0., *args, **kwargs):
        super().__init__(*args, **kwargs)

        mesh = mesh_descriptor.mesh
        if isinstance(mesh, Meshes):
            self.mesh: Meshes = mesh.to(self.device)
        elif isinstance(mesh, str):
            self.mesh: Meshes = py3dio.load_objs_as_meshes([mesh], device=self.device)

        self.pytorch3d_module: RenderModule = pytorch3d_module
        self.sailenv_module: SailenvModule = sailenv_module

        self.mesh_name = mesh_descriptor.mesh_name

        self.mesh_descriptor = mesh_descriptor

        self.classifier = classifier
        self.epsilon = epsilon

        self.src_texture = self.mesh.textures.maps_padded()

        self.delta = torch.nn.Parameter(
            torch.zeros_like(self.src_texture)
        )
        self.register_parameter("delta", self.delta)

        if saliency_maps is not None:
            self.saliency_threshold = saliency_threshold
            saliency_maps = saliency_maps.clone()
            # rescale saliency maps to 0..1 range
            saliency_maps = (saliency_maps - saliency_maps.min()) / (saliency_maps.max() - saliency_maps.min())
            saliency_maps[saliency_maps < saliency_threshold] = 0.
            self.saliency_maps = saliency_maps
        else:
            self.saliency_maps = None

        # call update_textures to set the parameter as texture
        # or else it will not use it at first epoch
        self.update_textures()

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()
        self.pytorch3d_test_accuracy = pl.metrics.Accuracy()
        self.sailenv_test_accuracy = pl.metrics.Accuracy()

        self.accuracies = {}

    def parameters(self, recurse: bool = True):
        return iter([self.delta])

    def update_textures(self):
        new_maps = self.src_texture + self.delta
        self.mesh.textures.set_maps(new_maps.clamp_(0., 1.))

    def replace_texture_on_descriptor(self):
        for mat_name, new_tex in self.get_textures().items():
            self.mesh_descriptor.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

    def __get_render_module(self, module_name):
        if module_name == PYTORCH3D_MODULE_NAME:
            return self.pytorch3d_module
        elif module_name == SAILENV_MODULE_NAME:
            return self.sailenv_module
        else:
            raise ValueError("Invalid module name")

    def apply_input(self, render_input, render_module):
        """

        :param render_input: 1x5 tensor in the form (distance, camera_azim, camera_elev, lights_azim, lights_elev)
        :param module_name: which module to use, can be PYTORCH3D_MODULE_NAME or SAILENV_MODULE_NAME
        :return: None
        """


        distance, camera_azim, camera_elev = render_input[:3]
        render_module.look_at_mesh(distance, camera_azim, camera_elev)

        lights_azim, lights_elev = render_input[3:]
        render_module.set_lights_direction(lights_azim, lights_elev)

    def render(self, render_module):
        return render_module.render(self.mesh)

    def get_view2tex_map(self):
        return self.pytorch3d_module.get_view2tex_map(self.mesh)

    def get_textures(self):
        textures = self.mesh.textures.get_textures()
        return {
            tex_name: texture.cpu()
            for tex_name, texture in textures.items()
        }

    def render_batch(self, render_inputs, module_name=PYTORCH3D_MODULE_NAME):
        # we must render each image separately, batch rendering is broken for some reason
        images = []
        view2tex_maps = []
        render_module = self.__get_render_module(module_name)

        for render_input in render_inputs:
            self.apply_input(render_input, render_module)
            image = self.render(render_module)
            view2tex_map = self.get_view2tex_map()
            images.append(image)
            view2tex_maps.append(view2tex_map)

        return torch.cat(images, 0), torch.cat(view2tex_maps, 0)

    def __log_accuracy(self, accuracy, phase: str):
        acc = accuracy.compute()
        self.accuracies[f"{phase}_accuracy"] = acc
        self.log(f"{self.mesh_name}/{phase}_accuracy", acc, prog_bar=True)

        if self.current_epoch == 0 and phase == 'train':
            # save the accuracy before attack
            self.accuracies["before_attack"] = acc

    def __log_textures(self):
        for tex_name, tex in self.get_textures().items():
            self.logger.experiment.add_image(f"{self.mesh_name}/textures/{tex_name}", tex.permute(2, 0, 1),
                                             global_step=self.current_epoch)

    def __log_delta(self):
        self.logger.experiment.add_image(f"{self.mesh_name}/delta", self.delta.squeeze(0).permute(2, 0, 1),
                                         global_step=self.current_epoch)

    def __log_delta_measure(self):
        delta_norm = self.delta.norm()
        self.log(f"{self.mesh_name}/delta_norm", delta_norm, prog_bar=False)
        delta_n_pixels = (self.delta != 0.).sum().to(dtype=torch.float32) / self.delta.numel()
        self.log(f"{self.mesh_name}/delta_n_pixels", delta_n_pixels, prog_bar=False)

    def on_train_epoch_start(self):
        self.train_accuracy.reset()

    def training_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx, "train", PYTORCH3D_MODULE_NAME)
        self.train_accuracy(predictions, targets)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.update_textures()
        pass

    def on_train_epoch_end(self, outputs):
        self.__log_accuracy(self.train_accuracy, "train")
        self.__log_textures()
        self.__log_delta()
        self.__log_delta_measure()

        if self.train_accuracy.compute() == 0.:
            self.trainer.should_stop = True

    def on_validation_epoch_start(self) -> None:
        self.replace_texture_on_descriptor()
        self.sailenv_module.spawn_obj(self.mesh_descriptor)
        self.valid_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx, "val", SAILENV_MODULE_NAME)
        self.valid_accuracy(predictions, targets)
        return total_loss

    def on_validation_epoch_end(self):
        self.__log_accuracy(self.valid_accuracy, "validation")
        self.sailenv_module.despawn_obj()

    def on_test_epoch_start(self) -> None:
        self.replace_texture_on_descriptor()
        self.sailenv_module.spawn_obj(self.mesh_descriptor)
        self.pytorch3d_test_accuracy.reset()
        self.sailenv_test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch, batch_idx, "test", PYTORCH3D_MODULE_NAME)
        self.pytorch3d_test_accuracy(predictions, targets)

        total_loss, predictions, targets = self.handle_batch(batch, batch_idx, "test", SAILENV_MODULE_NAME)
        self.sailenv_test_accuracy(predictions, targets)
        return total_loss

    def on_test_epoch_end(self):
        self.__log_accuracy(self.pytorch3d_test_accuracy, "pytorch3d_test")
        self.__log_accuracy(self.sailenv_test_accuracy, "sailenv_test")
        self.sailenv_module.despawn_obj()

    def to(self, device):
        self.mesh = self.mesh.to(device)
        self.mesh.textures = self.mesh.textures.to(device)
        self.src_texture = self.src_texture.to(device)
        self.pytorch3d_module.to(device)
        self.sailenv_module.to(device)
        self.classifier.to(device)
        super().to(device)

    def cuda(self, deviceId=None):
        self.to(deviceId)
        super().cuda(deviceId)

    def cpu(self):
        self.to('cpu')
        super().cpu()

    def register_hooks(self, images, batch_idx):
        pass

    def handle_batch(self, batch, batch_idx, phase="train", module_name=PYTORCH3D_MODULE_NAME):
        """
        N batch size, WxH view size
        :param batch:
        :param batch_idx:
        :return:
        """
        render_inputs, targets = batch
        images, view2tex_maps = self.render_batch(render_inputs, module_name)

        if phase == "train":
            self.register_hooks(images, batch_idx)

        images_grid = Viewer3D.make_grid(images)
        self.logger.experiment.add_image(f"{self.mesh_name}/{phase}_{module_name}_batch{batch_idx}",
                                         images_grid.permute((2, 0, 1)), global_step=self.current_epoch)

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
