from typing import Any, Union

import torch

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
                "name": f"delta_texture_{idx}",
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
                param.data.add_(perturbation)

        return loss


class FGSMAttack(SaifoolerAttack):
    def configure_optimizers(self):
        return FGSMOptimizer(self.parameters(), self.epsilon)

