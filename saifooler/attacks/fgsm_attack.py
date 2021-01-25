from typing import Any, Union

import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
import torch.nn.functional as F

import pytorch3d.io as py3dio
from torch.nn import CrossEntropyLoss

from saifooler.attacks.attack import SaifoolerAttack
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


class FGSMAttack(SaifoolerAttack):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def handle_mini_batch(self, mini_batch, mini_batch_idx):
        render_inputs, targets = mini_batch

        images = self.render_batch(render_inputs)

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
        self.logger.experiment.add_image(f"{self.mesh_name}/pytorch3d_batch{mini_batch_idx}", images_grid.permute((2,0,1)))

        return loss, classes_predicted

    def configure_optimizers(self):
        return FGSMOptimizer(self.parameters(), self.epsilon)

