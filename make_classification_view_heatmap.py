import json

import torch
import pytorch_lightning as pl
from sailenv.agent import Agent
from torchvision import models
import sys
import argparse

from saifooler.data_modules.multiple_viewpoints_module import MultipleViewModule
from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.pgd_attack import PGDAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier
from saifooler.render.sailenv_module import SailenvModule
from tqdm import tqdm

from sailenv_manager import SAILenvManager


#def tqdm(*args, **kwargs):
#    return args[0]


import seaborn as sns
sns.set_theme()

from pytorch_lightning.loggers import TensorBoardLogger

from saifooler.saliency.saliency_estimator import SaliencyEstimator
from saifooler.utils import greyscale_heatmap
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

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
parser.add_argument('--unitytest', metavar='unity', type=bool,
                    default=False, help="Wheter to test on unity." )


def create_unity_agent(args):
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

    return agent


def classify_module(render_module, classifier, mesh, data_module):

    inputs_list = []
    classes_predicted_list = []
    confidences_list = []

    for batch in tqdm(data_module.test_dataloader(), position=1, desc="Batch"):
        render_inputs, targets = [x.to(device) for x in batch]
        images = []
        for render_input in render_inputs:
            distance, camera_azim, camera_elev = render_input[:3]
            render_module.look_at_mesh(distance, camera_azim, camera_elev)

            lights_azim, lights_elev = render_input[3:]
            render_module.set_lights_direction(lights_azim, lights_elev)

            image = render_module.render(mesh)
            images.append(image)

        images = torch.cat(images, 0)
        # classify images and extract class predictions
        class_tensors = classifier.classify(images)
        confidences, classes_predicted = class_tensors.max(1, keepdim=True)
        confidences[classes_predicted != targets] = 0.
        confidences_list.append(confidences)
        classes_predicted_list.append(classes_predicted)
        inputs_list.append(render_inputs)

    confidences = torch.cat(confidences_list, 0)
    classes_predicted = torch.cat(classes_predicted_list, 0)
    inputs = torch.cat(inputs_list, 0)

    plt.figure(figsize=(7, 7))
    plt.imshow(images[0].cpu().squeeze(0).numpy())
    plt.title("unity")
    plt.show()

    return torch.cat((inputs.to(confidences), classes_predicted, confidences), 1)


def classify_and_draw_heatmap(render_module, classifier, mesh, data_module, prefix):
    classify_out = classify_module(render_module, classifier, mesh, data_module).cpu()

    classify_out = classify_out.reshape(
        (
            data_module.orientation_azim_steps,
            data_module.orientation_elev_steps,
            data_module.light_azim_steps * data_module.light_elev_steps,
            -1
        )
    )

    scores_grid = classify_out[..., 6].mean(dim=2)
    xticks = [f"{x:.1f}" for x in classify_out[..., 1, ...].unique().tolist()]
    yticks = [f"{x:.1f}" for x in classify_out[..., 2, ...].unique().tolist()]

    plt.figure()
    sns.heatmap(scores_grid.transpose(1, 0), xticklabels=xticks,
                yticklabels=yticks, vmin=0.)

    plt.grid()
    plt.title(f"Mesh: {mesh_name}")
    plt.savefig(f"./heatmaps/new/{prefix}_{mesh_name}.png")




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

    test_on_unity = args.unitytest

    if test_on_unity:
        agent = create_unity_agent(args)

    classifier = ImageNetClassifier(used_model)
    classifier.to(device)
    used_model.eval()

    render_module = RenderModule()
    render_module.to(device)

    sailenv_module = SailenvModule(agent, render_module.lights)
    sailenv_module.to(device)

    skip_meshes_before = 1

    manager = SAILenvManager(sailenv_home="C:\\Users\\enric\\wkspaces\\sailab_lve\\Build\\windows")

    # manager.start()

    with torch.no_grad():
        i = 0
        for mesh_name, mesh_def in tqdm(meshes_def.items(), position=0, desc="Mesh"):
            i += 1
            if i - 1 < skip_meshes_before:
                continue

            mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
            distance = mesh_def["distance"]
            orientation_elev_range = mesh_def["orientation_elev_range"]
            mesh_descriptor = MeshDescriptor(mesh_path)
            data_module = MultipleViewModule(
                target_class, distance,
                orientation_elev_steps=20,
                orientation_azim_steps=10,
                orientation_elev_range=orientation_elev_range,
                light_azim_steps=1,
                light_elev_steps=1,
                batch_size=45)
            data_module.setup()

            mesh_descriptor.mesh = mesh_descriptor.mesh.to(device)
            mesh_descriptor.mesh.textures = mesh_descriptor.mesh.textures.to(device)

            if test_on_unity:
                original_zip_path = mesh_descriptor.save_to_zip()
                sailenv_module.spawn_obj(mesh_descriptor)
                classify_and_draw_heatmap(sailenv_module, classifier, mesh_descriptor.mesh, data_module, "sailenv")
                sailenv_module.despawn_obj()

            classify_and_draw_heatmap(render_module, classifier, mesh_descriptor.mesh, data_module, "pytorch3d")
    # manager.stop()

