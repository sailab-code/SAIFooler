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
from pathlib import Path

from utils import input_transforms, imshow, visualize_model, idx2label

# import dataset as dataloader

use_cuda = True
dev = 0
assert type(dev) == int
batch = 6

if __name__ == '__main__':


    folder_dataloader = torchvision.datasets.ImageFolder(root='dataset/', transform=input_transforms)



    data_loader = torch.utils.data.DataLoader(folder_dataloader,
                                              batch_size=batch,
                                              shuffle=True,
                                              num_workers=1)

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # inception = models.inception_v3(pretrained=True).to(device)

    mobilenet = models.mobilenet_v2(pretrained=True).to(device)

    used_model = mobilenet

    used_model.eval()

    class_names = data_loader.dataset.classes

    # Make a grid from batch
    inputs, classes = next(iter(data_loader))
    print(f"Ground truth classes: \t {[class_names[x] for x in classes]}")
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    pred = used_model(inputs.to(device))

    pred_class = visualize_model(used_model, inputs, classes, device, idx2label, num_images=batch)
    print(f"Predicted classes: \t \t {pred_class}")
