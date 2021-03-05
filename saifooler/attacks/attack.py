import abc
import traceback
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
import seaborn as sns
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

PYTORCH3D_MODULE_NAME = "pytorch3d"
SAILENV_MODULE_NAME = "sailenv"

TRAIN_PHASE = "train"
VAL_PHASE = "validation"
TEST_PHASE = "test"


class SaifoolerAttack(pl.LightningModule, metaclass=abc.ABCMeta):


    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh_descriptor: MeshDescriptor, pytorch3d_module, sailenv_module, classifier, epsilon, datamodule,
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

        self.datamodule = datamodule

        self.batch_handlers = {
            TRAIN_PHASE: self.handle_train_batch,
            VAL_PHASE: self.handle_val_batch,
            TEST_PHASE: self.handle_test_batch
        }

        # call update_textures to set the parameter as texture
        # or else it will not use it at first epoch
        self.apply_delta_to_textures()

        self.pytorch3d_accuracy = pl.metrics.Accuracy()
        self.sailenv_accuracy = pl.metrics.Accuracy()

        self.accuracies = {}

        self.p3d_heatmap_data = {}
        self.sailenv_heatmap_data = {}

    def random_initialize_delta(self):
        torch.nn.init.uniform_(self.delta, -self.epsilon, self.epsilon)
        self.apply_delta_to_textures()

    def parameters(self, recurse: bool = True):
        return iter([self.delta])

    def reset_textures(self):
        self.mesh.textures.set_maps(self.src_texture.clamp(0., 1.))

    def apply_delta_to_textures(self):
        new_maps = self.src_texture + self.delta
        self.mesh.textures.set_maps(new_maps.clamp(0., 1.))

    def replace_texture_on_descriptor(self):
        for mat_name, new_tex in self.get_textures().items():
            self.mesh_descriptor.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

    def rescale_textures_on_descriptor(self, scale):
        for mat_name, new_tex in self.get_textures().items():
            self.mesh_descriptor.rescale_texture(mat_name, "albedo", scale)

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
        delta = self.delta.squeeze(0).permute(2, 0, 1)

        # add 0.5 to show also negative values
        delta = delta + 0.5


        self.logger.experiment.add_image(f"{self.mesh_name}/delta", delta,
                                         global_step=self.current_epoch)

    def __log_delta_measure(self):
        delta_norm = self.delta.norm(p=float('inf'))
        self.log(f"{self.mesh_name}/delta_norm", delta_norm, prog_bar=False)
        delta_n_pixels = (self.delta != 0.).sum().to(dtype=torch.float32) / self.delta.numel()
        self.log(f"{self.mesh_name}/delta_n_pixels", delta_n_pixels, prog_bar=False)

    def on_train_epoch_start(self):
        self.pytorch3d_accuracy.reset()

    def training_step(self, batch, batch_idx):
        batch_out = self.handle_train_batch(batch, batch_idx)
        loss, predicted, targets, _ = batch_out[PYTORCH3D_MODULE_NAME]
        self.pytorch3d_accuracy(predicted, targets)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.apply_delta_to_textures()
        pass

    def on_train_epoch_end(self, outputs):
        self.__log_accuracy(self.pytorch3d_accuracy, "pytorch3d")
        self.__log_delta_measure()

        if self.pytorch3d_accuracy.compute() == 0.:
            self.trainer.should_stop = True

    def on_validation_epoch_start(self) -> None:
        self.replace_texture_on_descriptor()
        self.sailenv_module.spawn_obj(self.mesh_descriptor)
        self.pytorch3d_accuracy.reset()
        self.sailenv_accuracy.reset()

        self.__reset_heatmap_data()

    def validation_step(self, batch, batch_idx):

        batch_out = self.handle_val_batch(batch, batch_idx)

        sailenv_loss, _, _, _ = batch_out[SAILENV_MODULE_NAME]
        return sailenv_loss

    def on_validation_epoch_end(self):
        self.__log_textures()
        self.__log_delta()
        self.__log_accuracy(self.sailenv_accuracy, "sailenv")
        self.__log_heatmap(self.p3d_heatmap_data, "pytorch3d", "Accuracy on PyTorch3D")
        self.__log_heatmap(self.sailenv_heatmap_data, "sailenv", "Accuracy on SAILenv")
        self.sailenv_module.despawn_obj()

    def __reset_heatmap_data(self):
        self.p3d_heatmap_data = {
            "inputs": [],
            "scores": []
        }

        self.sailenv_heatmap_data = {
            "inputs": [],
            "scores": []
        }

    def __log_heatmap(self, heatmap_data, log_name, title=""):
        inputs = torch.cat(heatmap_data['inputs'], 0)
        predictions = torch.cat(heatmap_data['scores'], 0)

        grid = torch.cat((inputs, predictions), 1)

        grid = grid.reshape(
            (
                self.datamodule.orientation_azim_steps,
                self.datamodule.orientation_elev_steps,
                self.datamodule.light_azim_steps * self.datamodule.light_elev_steps,
                -1
            )
        )

        scores_grid = grid[..., 5].mean(dim=2)
        xticks = [f"{x:.1f}" for x in grid[..., 1, ...].unique().tolist()]
        yticks = [f"{x:.1f}" for x in grid[..., 2, ...].unique().tolist()[::-1]]

        plt.figure()
        fig = sns.heatmap(scores_grid.transpose(1, 0).flipud().cpu().clamp(min=1e-12).numpy(),
                          xticklabels=xticks, yticklabels=yticks, vmin=0., vmax=30.)
        # plt.grid()
        plt.title(title)
        
        self.logger.experiment.add_figure(f"{self.mesh_name}/accuracy_heatmap/{log_name}", fig.get_figure(),
                                          global_step=self.current_epoch)

    def __log_image(self, image, log_name):
        self.logger.experiment.add_image(log_name, image.permute((2, 0, 1)), global_step=self.current_epoch)

    def on_test_epoch_start(self) -> None:
        self.replace_texture_on_descriptor()
        self.sailenv_module.spawn_obj(self.mesh_descriptor)
        self.pytorch3d_accuracy.reset()
        self.sailenv_accuracy.reset()

        self.__reset_heatmap_data()

    def test_step(self, batch, batch_idx):
        batch_out = self.handle_test_batch(batch, batch_idx)

        sailenv_loss, _, _, _ = batch_out[SAILENV_MODULE_NAME]

        return sailenv_loss

    def on_test_epoch_end(self):
        self.__log_accuracy(self.pytorch3d_accuracy, "pytorch3d")
        self.__log_accuracy(self.sailenv_accuracy, "sailenv")
        
        self.__log_heatmap(self.p3d_heatmap_data, "pytorch3d", "Accuracy on PyTorch3D")
        self.__log_heatmap(self.sailenv_heatmap_data, "sailenv", "Accuracy on SAILenv")
        
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

    def __classify(self, images):
        # classify images and extract class predictions
        class_tensor = self.classifier.classify(images)
        return class_tensor

    def __compute_loss(self, class_tensor, classes_predicted, targets, **kwargs):
        # mask images on which the prediction was wrong
        loss_targets = targets.clone()
        loss_targets[classes_predicted != targets] = -1
        loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=-1)

        # compute CrossEntropyLoss
        ce_loss = loss_fn(class_tensor, loss_targets.squeeze(1))
        return ce_loss

    def __render_and_classify(self, render_inputs, targets, module_name):
        images, _ = self.render_batch(render_inputs, module_name)
        class_tensor = self.__classify(images)
        scores, classes_predicted = class_tensor.max(1, keepdim=True)

        loss = self.__compute_loss(class_tensor, classes_predicted, targets)

        return loss, images, classes_predicted, targets, scores

    def handle_train_batch(self, batch, batch_idx):
        render_inputs, targets = batch

        loss, images, classes_predicted, targets, scores = self.__render_and_classify(
            render_inputs, targets, PYTORCH3D_MODULE_NAME
        )

        self.pytorch3d_accuracy(classes_predicted, targets)

        self.register_hooks(images, batch_idx)

        return {
            PYTORCH3D_MODULE_NAME: (loss, classes_predicted, targets, scores)
        }

    def handle_val_batch(self, batch, batch_idx):
        render_inputs, targets = batch
        p3d_loss, p3d_images, p3d_predicted, p3d_targets, p3d_scores = self.__render_and_classify(
            render_inputs, targets, PYTORCH3D_MODULE_NAME
        )
        self.p3d_heatmap_data["inputs"].append(render_inputs)
        self.p3d_heatmap_data["scores"].append(p3d_scores)

        self.pytorch3d_accuracy(p3d_predicted, targets)

        images_grid = Viewer3D.make_grid(p3d_images)
        self.__log_image(images_grid, f"{self.mesh_name}/{PYTORCH3D_MODULE_NAME}_batch{batch_idx}")

        sailenv_loss, sailenv_images, sailenv_predicted, sailenv_targets, sailenv_scores = self.__render_and_classify(
            render_inputs, targets, SAILENV_MODULE_NAME
        )

        images_grid = Viewer3D.make_grid(sailenv_images)
        self.__log_image(images_grid, f"{self.mesh_name}/{SAILENV_MODULE_NAME}_batch{batch_idx}")

        self.sailenv_heatmap_data["inputs"].append(render_inputs)
        self.sailenv_heatmap_data["scores"].append(sailenv_scores)
        self.sailenv_accuracy(sailenv_predicted, targets)
        return {
            PYTORCH3D_MODULE_NAME: (p3d_loss, p3d_predicted, p3d_targets, p3d_scores),
            SAILENV_MODULE_NAME: (sailenv_loss, sailenv_predicted, sailenv_targets, sailenv_scores)
        }

    def handle_test_batch(self, batch, batch_idx):
        return self.handle_val_batch(batch, batch_idx)

    def handle_batch(self, batch, batch_idx, phase="train"):
        """
        N batch size, WxH view size
        :param batch:
        :param batch_idx:
        :return:
        """

        handler = self.batch_handlers[phase]
        return handler(batch, batch_idx)

    def saliency_hook(self, grad: torch.Tensor, offset):
        try:
            if self.saliency_maps is None:
                return grad

            n = grad.norm()
            s = self.saliency_maps[offset:offset+grad.shape[0]]
            grad = grad * s

            grad = grad / grad.norm() * n

            return grad
        except Exception:
            print("error")
            traceback.print_exc()

    def __generate_saliency_hook(self, offset):
        f = self.saliency_hook

        def hook(grad):
            return f(grad, offset)

        return hook

    def register_hooks(self, images, batch_idx):
        offset = batch_idx * images.shape[0]  # assume all batches have same shape

        images.register_hook(self.__generate_saliency_hook(offset))

    @abc.abstractmethod
    def configure_optimizers(self):
        pass
