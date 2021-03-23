import json
import os
from time import sleep

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
from saifooler.render.sailenv_module import SailenvModule

from pytorch_lightning.loggers import TensorBoardLogger
from saifooler.utils import SummaryWriter
from saifooler.saliency.saliency_estimator import SaliencyEstimator
# from saifooler.utils import greyscale_heatmap
# import torchvision.transforms.functional as TF
# from PIL import Image, ImageEnhance
# import matplotlib.pyplot as plt
from itertools import product

from saifooler_pgd_attack import experiment
from sailenv_manager import SAILenvManager

parser = argparse.ArgumentParser(description="Settings for PGD Attack to obj textures")
parser.add_argument('--meshes_definition', metavar='meshes_definition', type=str,
                    required=True,
                    help="Path to a json file which defines the meshes to be attacked. "
                         "The file must contain an object with the following structure."
                         '{ "<obj_name>": { "path": "<obj_dir>", "distance": "<viewing distance>", '
                         '"target_class": <imagenet_class_id> },...} See meshes_definition.example.json for an example.'
                    )
# parser.add_argument('--eps', metavar="epsilon", type=float,
#                     required=True,
#                     help="Epsilon of the PGD attack")
# parser.add_argument('--alpha', metavar="alpha", type=float,
#                     required=True,
#                     help="Alpha of the PGD attack")
# parser.add_argument('--saliency', action="store_true",
#                     help="Wheter to use saliency for attack")
# parser.add_argument('--classifier', metavar="classifier", type=str,
#                     required=True,
#                     help="The classifier to be attacked. Choose between inception and mobilenet.")
parser.add_argument('--cuda', metavar="cuda", type=bool,
                    default=True, help="Set to true if you want to use GPU for training")
parser.add_argument('--device', metavar="device", type=int,
                    default=0, help="What GPU to be used for training")
parser.add_argument('--host', metavar='host', type=str,
                    default="127.0.0.1", help="Host on which SAILenv server resides")
parser.add_argument('--port', metavar='port', type=int,
                    default=8085, help="Port on which SAILenv server resides")


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
    agent.change_main_camera_clear_flags(255, 255, 255)

    #agent.change_scene("object_view/scene")
    return agent


if __name__ == '__main__':
    args = parser.parse_args()

    meshes_json_path = args.meshes_definition

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    # hyperparams

    EPS = [0.1]
    ALPHA = [0.01]
    CLASSIFIER = ["inception", "mobilenet"]
    SALIENCY = [False]
    SALIENCY_THRESH = [0.05]
    TEXTURE_RESCALE = [0.33]
    BACKGROUNDS = ["white", "black", "toilet", "room"]

    sailenv_manager = SAILenvManager(sailenv_home="/media/users/tiezzi/Projects/inProgress/SAILenv",
                                     port=str(args.port))

    for mesh_name, mesh_def in meshes_def.items():

        sailenv_manager.start()
        sleep(5)
        sailenv_manager.change_scene("object_view/scene")
        sleep(10)

        mesh_def["name"] = mesh_name

        for eps_, alpha_, classifier_, saliency_, tex_scale_, background_ in product(EPS, ALPHA, CLASSIFIER, SALIENCY,
                                                                        TEXTURE_RESCALE, BACKGROUNDS):

            exp_name_base = f"eps_{eps_}__alpha_{alpha_}__model_{classifier_}_saliency_{saliency_}_texscale_{tex_scale_}_background_{background_}"

            params_dict = {"eps": eps_,
                           "alpha": alpha_,
                           "model": classifier_,
                           "saliency": saliency_,
                           "texture_rescale": tex_scale_
                           }

            mesh_def["background"] = f"./backgrounds/{background_}.png"

            if saliency_:
                saliency_thresh_ = SALIENCY_THRESH
            else:
                saliency_thresh_ = [-1]

            for s_th_ in saliency_thresh_:

                if saliency_:
                    exp_name = exp_name_base + f"_saliency_thresh_{s_th_}"
                    params_dict["saliency_threshold"] = s_th_
                else:
                    exp_name = exp_name_base

                model_name = classifier_

                log_dir = f"bg_test_2/{mesh_name}"
                full_path_log_dir = os.path.join(f"{log_dir}/pgd_attack", exp_name)

                if os.path.exists(full_path_log_dir):
                    print(f"Experiment {full_path_log_dir} already available...")
                    continue

                experiment(exp_name, mesh_def, params_dict, args, log_dir=log_dir,
                           switch_testdata=False)

        sailenv_manager.stop()
        sleep(10)