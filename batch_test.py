import json

import torch
import pytorch_lightning as pl
from pytorch3d.renderer import look_at_view_transform
from sailenv.agent import Agent
from torchvision import models
import sys
import argparse
import seaborn as sns

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.fgsm_attack import FGSMAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier
from saifooler.render.sailenv_module import SailenvModule

import pytorch3d.io as py3dio

from pytorch_lightning.loggers import TensorBoardLogger
if __name__ == '__main__':

    device = torch.device("cuda:0")
    used_model = models.inception_v3(pretrained=True).to(device)

    meshes_json_path = "./meshes_definition.example.json"

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    mesh_def = list(meshes_def.values())[0]

    mesh_path, target_class, elevation, distance = mesh_def["path"], mesh_def["target_class"], mesh_def["elevation"], mesh_def["distance"]

    data_module = OrientationDataModule(target_class, elevation, distance, 4)
    classifier = ImageNetClassifier(used_model)
    render_module = RenderModule()

    data_module.setup()
    inputs = data_module.inputs

    distances, elevations, azimuths = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    cam_sets = look_at_view_transform(distances, elevations, azimuths)
    render_module.update_camera(cam_sets[0], cam_sets[1])
    render_module.to(device)

    mesh = py3dio.load_objs_as_meshes([mesh_path+"/remote_controller.obj"], device=device)
    mesh.textures.to(device)

    mesh_ext = mesh.extend(4)
    x = render_module.render(mesh)
    print(x)



