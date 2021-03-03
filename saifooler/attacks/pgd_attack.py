from typing import Any, Union

import torch
from pytorch3d.structures import Meshes

from torch.nn import CrossEntropyLoss

from saifooler.attacks.attack import SaifoolerAttack
from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.viewers.viewer import Viewer3D

import traceback

class PGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, alpha, epsilon, src_texture):

        if alpha < 0.:
            raise ValueError("alpha must be a non-negative value")

        if epsilon < 0.:
            raise ValueError("epsilon must be a non-negative value.")

        params_list = []
        for idx, param in enumerate(params):
            param_dict = {
                "params": param,
                "name": f"texture {idx}",
                "src_tex": src_texture,
                "eps": epsilon,
                "alpha": alpha
            }
            params_list.append(param_dict)

        defaults = {}

        super().__init__(params_list, defaults)

    @staticmethod
    def _norms(t):
        # compute norms over all but the first dimension
        # taken from https://adversarial-ml-tutorial.org/adversarial_examples/
        return t.view(t.shape[0], -1).norm(dim=1)[:, None, None, None]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            alpha = group["alpha"]
            src_tex = group["src_tex"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                p_grad = param.grad
                p_data = param.data.clone()

                perturbation = alpha * p_grad.sign()
                p_data += perturbation

                # clamp linf norm to eps
                p_data = p_data.clamp(min=-eps, max=eps)

                # clip src_tex + param between [0,1]
                p_data = src_tex - torch.clamp(src_tex + p_data == 0., 1.)
                param.data = p_data

        return loss


class PGDAttack(SaifoolerAttack):
    def __init__(self, mesh_descriptor: MeshDescriptor, pytorch3d_module, sailenv_module, classifier, epsilon, alpha, datamodule, *args, **kwargs):
        super().__init__(mesh_descriptor, pytorch3d_module, sailenv_module, classifier, epsilon, datamodule, *args, **kwargs)
        self.alpha = alpha

    def configure_optimizers(self):
        return PGDOptimizer(self.parameters(), self.alpha, self.epsilon, self.src_texture)
