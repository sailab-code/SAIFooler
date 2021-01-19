from typing import Any, Union

import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import torch.nn.functional as F

import pytorch3d.io as py3dio
from torch.nn import CrossEntropyLoss

from saifooler.viewers.viewer import Viewer3D


class FGSMOptimizer(torch.optim.Optimizer):
    def __init__(self, params, epsilon):

        if epsilon < 0.:
            raise ValueError("epsilon must be a non-negative value.")

        params_list = []
        for idx, param in enumerate(params):
            param_dict = {
                "params": param,
                "name": f"texture {idx}",
                "eps": epsilon
            }
            params_list.append(param_dict)

        defaults = {}

        super().__init__(params_list, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad_sign = param.grad.sign()
                perturbation = grad_sign * eps
                param.data.add_(perturbation).clamp_(0., 1.)

        return loss


class FGSMAttack(pl.LightningModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, mesh: Union[str, Meshes], render_module, classifier, epsilon, mesh_name="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(mesh, Meshes):
            self.mesh: Meshes = mesh.to(self.device)
        elif isinstance(mesh, str):
            self.mesh: Meshes = py3dio.load_objs_as_meshes([mesh], device=self.device)

        self.render_module = render_module
        self.mesh_name = mesh_name

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

    def handle_batch(self, batch):
        render_inputs, targets = batch

        """# apply batch of inputs and render
        self.apply_input(render_inputs[:, 0], render_inputs[:, 1], render_inputs[:, 2])
        images = self.render()

        # classify images and extract class predictions
        class_tensors = self.classifier.classify(images)
        _, classes_predicted = class_tensors.max(1, keepdim=True)

        # mask images on which the prediction was wrong
        targets[classes_predicted != targets] = -1
        loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=-1)

        # compute CrossEntropyLoss
        loss = loss_fn(class_tensors, targets.squeeze(1))
        loss = class_tensors[0,0]
        images_grid = Viewer3D.make_grid(images)
        self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d", images_grid.permute((2,0,1)))

        return loss, classes_predicted, targets"""

        images = []
        predictions = []

        total_loss = torch.tensor(0., device=self.device)

        # todo: see if operations can be batched
        for render_input, target in zip(render_inputs, targets):
            self.apply_input(*render_input)
            image = self.render()
            images.append(image.cpu())
            class_tensor = self.classifier.classify(image)
            _, class_predicted = class_tensor.max(1, keepdim=True)
            class_predicted = class_predicted.squeeze(0)
            predictions.append(class_predicted.cpu())
            # target = target.unsqueeze(0)

            current_loss = F.nll_loss(class_tensor, target, reduction="mean")
            del class_tensor
            del image
            # if prediction is already wrong skip attack for this input
            if class_predicted != target:
                current_loss = current_loss * 0

            total_loss = total_loss + current_loss

        # todo: when removing the for loop, remove also scaling factor below
        return total_loss / render_inputs.shape[0], torch.tensor(predictions, device=self.device), targets.squeeze(1)

    def configure_optimizers(self):
        return FGSMOptimizer(self.parameters(), self.epsilon)

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
        total_loss, predictions, targets = self.handle_batch(batch)
        self.train_accuracy(predictions, targets)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.update_textures()

    def on_train_epoch_end(self, outputs):
        self.__log_accuracy(self.train_accuracy, "train")

    def on_validation_epoch_start(self) -> None:
        self.valid_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch)
        self.valid_accuracy(predictions, targets)
        return total_loss

    def on_validation_epoch_end(self):
        self.__log_accuracy(self.valid_accuracy, "validation")

    def on_test_epoch_start(self) -> None:
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        total_loss, predictions, targets = self.handle_batch(batch)
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