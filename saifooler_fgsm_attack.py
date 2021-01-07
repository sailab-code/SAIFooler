import torch
import pytorch_lightning as pl
from torchvision import models
import sys

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


mesh_path = "./meshes/table_living_room/table_living_room.obj"

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

    render_module = RenderModule()
    classifier = ImageNetClassifier(used_model)
    data_module = OrientationDataModule(target_class, 10., 2., 30)
    attacker = FGSMAttack(mesh_path, render_module, classifier, epsilon)
    viewer = Viewer3D(attacker)

    # show model before training
    view_model(viewer)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        progress_bar_refresh_rate=0,
        gpus=1
    )

    print("Attack begin")
    trainer.fit(attacker, datamodule=data_module)
    print("Testing")
    trainer.test(attacker, datamodule=data_module)
    print("Attack end")

    # show model after training
    view_model(viewer)