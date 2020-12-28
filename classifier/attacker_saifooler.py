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
import sys
from utils import input_transforms, imshow, visualize_model, idx2label, fgsm_attack, build_attack, cls2label, \
    label2idx, imshow_transform_notensor
from pathlib import Path

from saifooler.texture_module import TextureModule
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# import dataset as dataloader

use_cuda = True
dev = 0
assert type(dev) == int

epsilons = [0, .05, .1, .15, .2, .25, .3]


class MeshDataModule(pl.LightningDataModule):
    def __init__(self, target_class, elev=0., distance=0.):
        super().__init__()
        self.target_class = target_class
        self.elev = elev
        self.distance = distance

    def train_dataloader(self):
        orientations = torch.arange(0., 360., 60.)

        inps = torch.zeros((orientations.shape[0], 3))
        inps[:,0], inps[:,1], inps[:,2] = self.distance, self.elev, orientations

        targets = torch.full_like(orientations, self.target_class)
        dset = TensorDataset(inps, targets)
        return DataLoader(dset, batch_size=1)




if __name__ == '__main__':

    filter_classes = ["table_living_room"]

    used_model_id = "mobilenet"
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    if used_model_id == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif used_model_id == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    used_model.eval()

    texture_module = TextureModule(mesh_path="../meshes/table_living_room/table_living_room.obj")
    texture_module.to(device)

    data_loader = MeshDataModule(target_class=532, elev=25, distance=2.5).train_dataloader()

    accuracies = []
    examples = []

    class_names = {532: "table_living_room"}
    imagenet_label_tensor = torch.tensor([532])
    # Make a grid from batch
    for inputs, classes in iter(data_loader):
        batch = inputs.shape[0]
        inputs = inputs.squeeze(0)

        images = texture_module.render(inputs)
        images = images.permute(2,0,1).unsqueeze(0)

        print(f"Ground truth classes: \t {[class_names[int(x)] for x in classes]}")
        out = torchvision.utils.make_grid(images)

        imshow(out.cpu().detach(), title=[class_names[int(x)] for x in classes])

        pred = used_model(images.to(device))
        top_3_val, top3_idx = torch.topk(pred, 3, dim=1)
        #pred_class = visualize_model(used_model, inputs, classes, device, idx2label, num_images=batch,
        #                            filter_classes=filter_classes, imagenet_label_tensor=imagenet_label_tensor)
        print(f"Predicted classes: \t \t {top_3_val}, {top3_idx}")

    """# Run test for each epsilon
    for eps in epsilons:
        acc, ex = build_attack(used_model, device, data_loader, eps, imaget_label_tensor)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))

            plt.imshow(imshow_transform_notensor(ex))
    plt.tight_layout()
    plt.show()"""
