import json

import torch
import pytorch_lightning as pl
from sailenv.agent import Agent
from torchvision import models
import sys
import argparse
import seaborn as sns

from saifooler.data_modules.multiple_viewpoints_module import MultipleViewModule
from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.pgd_attack import PGDAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier
from saifooler.render.unity_evaluator import SailenvModule

from pytorch_lightning.loggers import TensorBoardLogger

from saifooler.saliency.saliency_estimator import SaliencyEstimator
from saifooler.utils import greyscale_heatmap
from saifooler.viewers.viewer import Viewer3D
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance

import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description="Settings for PGD Attack to obj textures")
parser.add_argument('--meshes_definition', metavar='meshes_definition', type=str,
                    required=True,
                    help="Path to a json file which defines the meshes to be attacked. "
                         "The file must contain an object with the following structure."
                         '{ "<obj_name>": { "path": "<obj_dir>", "distance": "<viewing distance>", '
                         '"target_class": <imagenet_class_id> },...} See meshes_definition.example.json for an example.'
                    )
parser.add_argument('--classifier', metavar="classifier", type=str,
                    required=True,
                    help="The classifier to be attacked. Choose between inception and mobilenet.")
parser.add_argument('--cuda', metavar="cuda", type=bool,
                    default=True, help="Set to true if you want to use GPU for training")
parser.add_argument('--device', metavar="device", type=int,
                    default=0, help="What GPU to be used for training")
parser.add_argument('--host', metavar='host', type=str,
                    default="127.0.0.1", help="Host on which SAILenv server resides")
parser.add_argument('--port', metavar='port', type=int,
                    default=8085, help="Port on which SAILenv server resides")


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

    meshes_json_path = args.meshes_definition

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    logger = TensorBoardLogger("./logs/saliency_test")

    host = args.host
    port = args.port
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False,
                  width=224, height=224, host=host,
                  port=port, use_gzip=False)
    # put white background on unity scene
    agent.register()
    agent.change_main_camera_clear_flags(255, 255, 255)
    agent.change_scene("object_view/scene")

    for mesh_name, mesh_def in meshes_def.items():
        mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
        distance = mesh_def["distance"]
        mesh_descriptor = MeshDescriptor(mesh_path)
        original_zip_path = mesh_descriptor.save_to_zip()
        classifier = ImageNetClassifier(used_model)
        render_module = RenderModule()

        data_module = MultipleViewModule(
            target_class, distance,
            orientation_elev_steps=3,
            orientation_azim_steps=3,
            light_azim_steps=1,
            light_elev_steps=1,
            batch_size=30)
        data_module.setup()

        sailenv_module = SailenvModule(agent, original_zip_path,
                                       f"{mesh_name}/sailenv",
                                       data_module, classifier,
                                       render_module)

        saliency_estimator = SaliencyEstimator(
            mesh_descriptor.mesh,
            classifier,
            render_module,
            sailenv_module,
            data_module,
            use_cache=False
        )

        saliency_estimator.to(device)

        sailenv_module.spawn_obj()
        saliency_maps = saliency_estimator.estimate_saliency_map()
        sailenv_module.despawn_obj()

        heatmaps = greyscale_heatmap(saliency_maps.unsqueeze(0).unsqueeze(3))  # NxWxHx1 between 0..1

        red_heatmap = torch.zeros((*heatmaps.shape[1:-1], 3))
        red_heatmap[..., 0] = heatmaps[..., 0]

        heatmap_img = TF.to_pil_image(red_heatmap.permute(2, 0, 1))

        src_texture = mesh_descriptor.mesh.textures.maps_padded()
        src_tex_img = TF.to_pil_image(src_texture.squeeze(0).permute(2, 0, 1).cpu())

        blended = Image.blend(heatmap_img, src_tex_img, 0.1)
        brightness_enhance = ImageEnhance.Brightness(blended)
        blended = brightness_enhance.enhance(3.5)

        blended = TF.pil_to_tensor(blended).to(dtype=torch.float32) / 255
        blended = blended.permute(1, 2, 0)

        plt.figure()
        plt.imshow(blended.cpu().numpy())
        plt.show()



    agent.delete()
