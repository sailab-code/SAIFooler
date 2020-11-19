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
import json

input_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

def imshow(inp, title=None, ax=None):
    """Imshow for Tensor."""
    inp = imshow_transform(inp)
    if ax:
        ax.imshow(inp)
    else:
        plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

idx2label = []
cls2label = {}
with open("dataset/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}


def visualize_model(model, inputs, labels, device, class_names,  num_images=4):

    was_training = model.training
    model.eval()
    images_so_far = 0

    fig = plt.figure()
    axes = []

    with torch.no_grad():

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        top_3_val, top3_idx = torch.topk(outputs, 3, dim=1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            axes.append(fig.add_subplot(num_images//2, 2, images_so_far))
            axes[-1].axis('off')
            axes[-1].set_title(f'predicted: {class_names[preds[j]]}')

            for k in range(3):
                plt.text(0, -0.1 - k * 0.15, f'{class_names[top3_idx[j][k].data.cpu()]} : {top_3_val[j][k].cpu().data:.3f}', fontsize=8, transform=axes[-1].transAxes)

            plt.imshow(imshow_transform(inputs.cpu().data[j]))

            if images_so_far == num_images:
                model.train(mode=was_training)
        model.train(mode=was_training)
        fig.tight_layout()
        plt.show()

