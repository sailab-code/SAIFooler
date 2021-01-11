import torch
import pytorch_lightning as pl
from torchvision import models
import sys

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.fgsm_attack import FGSMAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier
import matplotlib.pyplot as plt

from saifooler.viewers.viewer import Viewer3D

use_cuda = False
dev = 0
model_name = "inception"
#model_name = "mobilenet"


mesh_path = "./meshes/toilet"

target_class = 532
epsilon = 0.66

views_module = OrientationDataModule(target_class, 45., 2., 4)
views_module.setup()

def view_model(viewer_):
    with torch.no_grad():
        viewer_.multi_view_grid(views_module.inputs)
        viewer_.textures()


if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    if model_name == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif model_name == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    mesh_object = MeshDescriptor(mesh_path)

    render_module = RenderModule()
    classifier = ImageNetClassifier(used_model)
    data_module = OrientationDataModule(target_class, 10., 2., 30)
    attacker = FGSMAttack(mesh_object.mesh, render_module, classifier, epsilon)
    viewer = Viewer3D(attacker)

    # show model before training
    view_model(viewer)

