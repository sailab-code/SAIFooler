import torch


def hook(grad):
    print(grad)

v = torch.tensor([3., 2., 1.], requires_grad=True)
x = 2 * v
x.register_hook(hook)
x = 1 * x
sum(2*x).backward()