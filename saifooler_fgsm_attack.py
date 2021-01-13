import json

import torch
import pytorch_lightning as pl
from torchvision import models
import sys
import argparse

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.fgsm_attack import FGSMAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier

from saifooler.viewers.viewer import Viewer3D



parser = argparse.ArgumentParser(description="Settings for FGSM Attack to obj textures")
parser.add_argument('--meshes_definition', metavar='meshes_definition', type=str,
                    required=True,
                    help="Path to a json file which defines the meshes to be attacked. "
                         "The file must contain an object with the following structure."
                         '{ "<obj_name>": { "path": "<obj_dir>", "distance": "<viewing distance>", '
                         '"target_class": <imagenet_class_id> },...} See meshes_definition.example.json for an example.'
                    )
parser.add_argument('--eps', metavar="epsilon", type=float,
                    required=True,
                    help="Epsilon of the FGSM attack")
parser.add_argument('--classifier', metavar="classivier", type=str,
                    required=True,
                    help="The classifier to be attacked. Choose between inception and mobilenet.")
parser.add_argument('--cuda', metavar="cuda", type=bool,
                    default=True, help="Set to true if you want to use GPU for training")
parser.add_argument('--device', metavar="device", type=int,
                    default=0, help="What GPU to be used for training")


def view_model(_viewer, _views_module):
    with torch.no_grad():
        _viewer.multi_view_grid(_views_module.inputs)
        _viewer.textures()


if __name__ == '__main__':
    args = parser.parse_args()

    dev, use_cuda = args.device, args.cuda
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model_name = args.classifier
    if model_name == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif model_name == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    epsilon = args.eps
    meshes_json_path = args.meshes_definition

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    classifier = ImageNetClassifier(used_model)
    render_module = RenderModule()

    for mesh_name, mesh_def in meshes_def.items():
        mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
        elevation, distance = mesh_def["elevation"], mesh_def["distance"]
        mesh_object = MeshDescriptor(mesh_path)

        data_module = OrientationDataModule(target_class, elevation, distance, 30)
        attacker = FGSMAttack(mesh_object.mesh, render_module, classifier, epsilon)
        attacker.to(device)

        viewer = Viewer3D(attacker)
        views_module = OrientationDataModule(target_class, 45., distance, 4)
        views_module.setup()

        # show model before training
        view_model(viewer, views_module)

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
        view_model(viewer, views_module)

        attacked_mesh = mesh_object.copy_to_dir(f"./meshes/attacks/{mesh_path}_attacked", overwrite=True)

        for mat_name, new_tex in attacker.get_textures().items():
            attacked_mesh.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

        attacked_mesh.save_to_zip()
