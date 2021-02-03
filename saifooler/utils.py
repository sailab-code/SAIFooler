import torch


def greyscale_heatmap(images):
    """

    :param images: shape NxWxH
    :return:
    """
    # rescale pixels in 0..1
    return images / images.max(1, keepdim=True)[0].max(2, keepdim=True)[0]

