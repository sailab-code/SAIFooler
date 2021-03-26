import json
import os
from time import sleep

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
from saifooler.render.sailenv_module import SailenvModule

from pytorch_lightning.loggers import TensorBoardLogger

from saifooler.saliency.saliency_estimator import SaliencyEstimator
from saifooler.utils import greyscale_heatmap, SummaryWriter, str2bool
from operator import itemgetter
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

import numpy as np

parser = argparse.ArgumentParser(description="Settings for PGD Attack to obj textures")
parser.add_argument('--meshes_definition', metavar='meshes_definition', type=str,
                    required=True,
                    help="Path to a json file which defines the meshes to be attacked. "
                         "The file must contain an object with the following structure."
                         '{ "<obj_name>": { "path": "<obj_dir>", "distance": "<viewing distance>", '
                         '"target_class": <imagenet_class_id> },...} See meshes_definition.example.json for an example.'
                    )
parser.add_argument('--texture-rescale', metavar="float", type=float,
                    required=False, default=1.0,
                    help="Scale factor of the albedo textures (defaults to 1., no rescale)")
parser.add_argument('--saliency-threshold', metavar="alpha", type=float,
                    default=0.02, help="Threshold for constructing saliency map")
parser.add_argument('--cuda', metavar="cuda", type=str2bool,
                    default=True, help="Set to true if you want to use GPU for training")
parser.add_argument('--device', metavar="device", type=int,
                    default=0, help="What GPU to be used for training")
parser.add_argument('--host', metavar='host', type=str,
                    default="127.0.0.1", help="Host on which SAILenv server resides")
parser.add_argument('--port', metavar='port', type=int,
                    default=8085, help="Port on which SAILenv server resides")
parser.add_argument('--classifier', metavar="classifier", type=str,
                    required=True,
                    help="The classifier to be attacked. Choose between inception and mobilenet.")


def generate_agent(args):
    host = args.host
    port = args.port
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False,
                  width=224, height=224, host=host,
                  port=port, use_gzip=False)
    agent.register()

    # put white background on unity scene
    agent.change_main_camera_clear_flags(0, 0, 0)

    agent.change_scene("object_view/scene")
    sleep(10)
    return agent


if __name__ == '__main__':
    args = parser.parse_args()

    dev, use_cuda = args.device, args.cuda
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    texture_rescale = args.texture_rescale
    saliency_threshold = args.saliency_threshold
    model_name = args.classifier

    meshes_json_path = args.meshes_definition

    if model_name == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif model_name == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)


    agent = generate_agent(args)

    for mesh_name, mesh_def in meshes_def.items():

        mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
        distance = mesh_def["distance"]
        orientation_elev_range = mesh_def.get("orientation_elev_range", [-90., 90.])

        datamodule = MultipleViewModule(
            target_class, distance,
            orientation_elev_range=orientation_elev_range,
            orientation_elev_steps=2,
            orientation_azim_steps=5,
            light_azim_range=(0., 0.),
            light_azim_steps=1,
            light_elev_range=(50., 90.),
            light_elev_steps=1,
            batch_size=10)

        datamodule.setup()

        mesh_descriptor = MeshDescriptor(mesh_path).copy_to_dir(f"./saliency_tmp/{mesh_name}",
                                                                overwrite=True)

        classifier = ImageNetClassifier(used_model)

        background = torch.tensor(np.array(Image.open(mesh_def["background"])), dtype=torch.float32) / 255

        render_module = RenderModule(background=background)
        sailenv_module = SailenvModule(agent, render_module.lights, background=background)

        for mat_name, mat in mesh_descriptor.textures_path.items():
            mesh_descriptor.rescale_texture(mat_name, "albedo", texture_rescale)

        saliency_estimator = SaliencyEstimator(
            mesh_descriptor,
            classifier,
            render_module,
            sailenv_module,
            datamodule
        )
        saliency_estimator.to(device)


        tex_saliencies, views = saliency_estimator.estimate_saliency_map(return_views=True)

        tex_saliencies = tex_saliencies[0]

        saliency_maps = tex_saliencies.clone()
        # rescale saliency maps to 0..1 range
        for idx, saliency_map in enumerate(saliency_maps):
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            saliency_map[saliency_map < saliency_threshold] = 0.
            saliency_maps[idx] = saliency_map

        tex_saliencies = saliency_maps.clone()

        views = views[0]



        render_inputs = datamodule.train_dataloader().dataset.tensors[0]

        imgs_path = f"./saliency_tmp/{mesh_name}/images/"
        os.makedirs(imgs_path, exist_ok=True)

        for idx, tex_saliency, view in zip(range(tex_saliencies.shape[0]), tex_saliencies, views):

            mesh_repl = mesh_descriptor.copy_to_dir(f"./saliency_tmp/{mesh_name}/repl", overwrite=True)

            tex_saliency = tex_saliency.T

            heatmaps = greyscale_heatmap(tex_saliency.unsqueeze(0))  # NxWxHx1 between 0..1

            red_heatmap = torch.zeros((*heatmaps.shape[1:], 4))
            red_heatmap[..., 0] = heatmaps[0]
            red_heatmap[..., 3] = heatmaps[0]

            heatmap_img = TF.to_pil_image(red_heatmap.permute(2, 0, 1))

            src_texture = mesh_repl.mesh.textures.maps_padded()
            tex_img = TF.to_pil_image(src_texture.squeeze(0).permute(2, 0, 1).cpu())

            heatmap_img = heatmap_img.convert('RGBA')

            tex_img = tex_img.convert('RGBA')

            blended = tex_img.copy()
            blended.paste(heatmap_img, (0, 0), heatmap_img)

            #blended = Image.blend(heatmap_img, tex_img, 0.9)
            """brightness_enhance = ImageEnhance.Brightness(blended)
            blended = brightness_enhance.enhance(3.5)"""

            blended = blended.convert('RGB')

            blended = TF.pil_to_tensor(blended).to(dtype=torch.float32) / 255
            blended = blended.permute(1, 2, 0)

            tex_plt = plt.figure()
            plt.imshow(blended.cpu().numpy())
            plt.title(f"Texture: $\\tau_S={saliency_threshold:.2f}$")
            plt.tight_layout()
            #plt.show()

            mesh_repl.mesh = mesh_repl.mesh.to(device)
            mesh_repl.mesh.textures.set_maps(blended.unsqueeze(0).to(device))

            tex_dict = {
                tex_name: texture.cpu()
                for tex_name, texture in mesh_repl.mesh.textures.get_textures().items()
            }

            for mat_name, new_tex in tex_dict.items():
                mesh_repl.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

            render_input = render_inputs[idx]

            sailenv_module.spawn_obj(mesh_repl)

            distance, camera_azim, camera_elev = render_input[:3]
            sailenv_module.look_at_mesh(distance, camera_azim, camera_elev)

            lights_azim, lights_elev = render_input[3:]
            sailenv_module.set_lights_direction(lights_azim, lights_elev)

            image = sailenv_module.render(mesh_descriptor.mesh)

            sailenv_module.despawn_obj()

            mesh_plt = plt.figure()
            plt.title(f"Object: $\\tau_S={saliency_threshold:.2f}$")
            plt.imshow(image.squeeze(0).cpu().numpy())
            plt.tight_layout()
            #plt.show()

            tex_plt.savefig(f"{imgs_path}/ts{saliency_threshold:.2f}_tex_{idx}.pdf")
            mesh_plt.savefig(f"{imgs_path}/ts{saliency_threshold:.2f}_mesh_{idx}.pdf")

            plt.close(tex_plt)
            plt.close(mesh_plt)





