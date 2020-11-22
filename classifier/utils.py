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
import torch.nn.functional as F

# check tranforms difference for inception vs other models
# https://discuss.pytorch.org/t/imagenet-example-with-inception-v3/1691/23


input_transforms = transforms.Compose([
    transforms.Resize(256),
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

def imshow_transform_notensor(inp):
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


idx2label = []
cls2label = {}
cls_id2label = {}
with open("dataset/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls_id2label = {k: class_idx[str(k)][1] for k in range(len(class_idx))}
    label2idx = {class_idx[str(k)][1]: k  for k in range(len(class_idx))}


def visualize_model(model, inputs, labels, device, class_names, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0

    fig = plt.figure()
    axes = []
    prediction_classes = []

    with torch.no_grad():

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        top_3_val, top3_idx = torch.topk(outputs, 3, dim=1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            if num_images > 1:
                axes.append(fig.add_subplot(num_images // 2, 2, images_so_far))
            else:
                axes.append(fig.add_subplot(num_images, 1, images_so_far))
            axes[-1].axis('off')
            axes[-1].set_title(f'predicted: {class_names[preds[j]]}')
            prediction_classes.append(class_names[preds[j]])
            for k in range(3):
                plt.text(0, -0.1 - k * 0.15,
                         f'{class_names[top3_idx[j][k].data.cpu()]} : {top_3_val[j][k].cpu().data:.3f}', fontsize=8,
                         transform=axes[-1].transAxes)

            plt.imshow(imshow_transform(inputs.cpu().data[j]))

            if images_so_far == num_images:
                model.train(mode=was_training)
        model.train(mode=was_training)
        fig.tight_layout()
        plt.show()
        return prediction_classes


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def build_attack(model, device, test_loader, epsilon, imagenet_class_idx, idx2label=None):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target_image_loader = data.to(device), target.to(device)

        target = imagenet_class_idx[target_image_loader.item()][1]

        # Set requires_grad attribute of tensor. Important for Attac
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # out = torchvision.utils.make_grid(data.cpu().detach())
        # class_names = test_loader.dataset.classes
        # imshow(out, title=[class_names[x] for x in target])

        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # visualize_model(model,  data, target, device, idx2label, num_images=1)

        # If the initial prediction is wrong, dont bother attacking, just move on
        #if init_pred.item() != target.item():
        if init_pred.item() != target:
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target_image_loader)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #if final_pred.item() == target.item():
        if final_pred.item() == target:
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
