from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from utils import input_transforms, imshow, visualize_model, idx2label, fgsm_attack, build_attack, cls2label, \
    label2idx

# import dataset as dataloader

use_cuda = True
dev = 0
assert type(dev) == int

epsilons = [0, .05, .1, .15, .2, .25, .3]

if __name__ == '__main__':
    folder_dataloader = torchvision.datasets.ImageFolder(root='dataset/', transform=input_transforms)
    # folder_dataloader = torchvision.datasets.ImageNet(root='dataset/', transform=input_transforms, split="val")

    data_loader = torch.utils.data.DataLoader(folder_dataloader,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    inception = models.inception_v3(pretrained=True).to(device)

    inception.eval()

    class_names = data_loader.dataset.classes

    imagenet_class_idx = {k: [class_names[k], label2idx[class_names[k]]] for k in range(len(class_names))}


    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = build_attack(inception, device, data_loader, eps, imagenet_class_idx )
        accuracies.append(acc)
        examples.append(ex)
