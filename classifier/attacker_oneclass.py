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

# import dataset as dataloader

use_cuda = True
dev = 0
assert type(dev) == int

epsilons = [0, .05, .1, .15, .2, .25, .3]

if __name__ == '__main__':

    filter_classes = ["toilet_seat"]

    used_model_id = "inception"

    def checkfun(args):
        az = Path(args)
        return az.parent.stem in filter_classes
    def ___find_classes(self, dir):
        return filter_classes, {c: i for i, c in enumerate(filter_classes)}


    torchvision.datasets.ImageFolder._find_classes = ___find_classes
    folder_dataloader = torchvision.datasets.ImageFolder(root='dataset_adv/', transform=input_transforms,
                                                         is_valid_file=checkfun, )

    data_loader = torch.utils.data.DataLoader(folder_dataloader,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    if used_model_id == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif used_model_id == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    used_model.eval()

    class_names = data_loader.dataset.classes

    imagenet_class_idx = {k: [class_names[k], label2idx[class_names[k]]] for k in range(len(class_names))}
    # correspondance imagefolder classes -> imagenet classes
    # {0: ['Persian_cat', 283], 1: ['goldfish', 1], 2: ['home_theater', 598], 3: ['hummingbird', 94],
    # 4: ['laptop', 620],  # 5: ['racket', 752], 6: ['remote_control', 761], 7: ['toilet_seat', 861]}

    imaget_label_tensor = [v[1] for k, v in imagenet_class_idx.items()]

    imaget_label_tensor = torch.tensor(imaget_label_tensor)
    accuracies = []
    examples = []

    # Run test for each epsilon
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
    plt.show()
