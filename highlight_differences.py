import sys
from PIL import Image, ImageEnhance
import torch

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    images_paths = sys.argv[1:]

    images = [Image.open(path) for path in images_paths]

    t_images = [TF.to_tensor(pic).permute((1, 2, 0)) for pic in images]

    diff = torch.where(t_images[0] != t_images[1], torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., 0.]]))
    #diff[images[0] != images[1]] = torch.tensor([1., 0., 0.])
    diff = diff.permute((2, 0, 1))
    diff = TF.to_pil_image(diff)

    blended = Image.blend(images[0], diff, 0.8)
    brightness_enhance = ImageEnhance.Brightness(blended)
    blended = brightness_enhance.enhance(3.5)

    blended.save("./out.png")

