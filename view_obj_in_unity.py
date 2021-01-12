import signal

import torch
import pytorch_lightning as pl
from torchvision import models
import sys

from saifooler.render.mesh_descriptor import MeshDescriptor
from saifooler.render.render_module import RenderModule
from saifooler.attacks.fgsm_attack import FGSMAttack
from saifooler.data_modules.orientation_data_module import OrientationDataModule
from saifooler.classifiers.image_net_classifier import ImageNetClassifier
from sailenv.agent import Agent
import matplotlib.pyplot as plt
import cv2

from saifooler.render.unity_render import UnityRender
from saifooler.viewers.viewer import Viewer3D

use_cuda = False
dev = 0
model_name = "inception"
#model_name = "mobilenet"


mesh_path = "./meshes/table_living_room_attacked"

target_class = 532
epsilon = 0.66

views_module = OrientationDataModule(target_class, 45., 2., 4)
views_module.setup()

signal.signal(signal.SIGINT, signal.default_int_handler)

def view_model(viewer_):
    with torch.no_grad():
        viewer_.multi_view_grid(views_module.inputs)
        viewer_.textures()

host = "127.0.0.1"

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

    print("Generating agent...")
    try:
        agent = Agent(depth_frame_active=False,
                      flow_frame_active=False,
                      object_frame_active=False,
                      main_frame_active=True,
                      category_frame_active=False, width=224, height=224, host=host, port=8085, use_gzip=False)
        agent.register()
        agent.change_main_camera_clear_flags(255, 255, 255)
        unity_render = UnityRender(agent, "./meshes/table_living_room_attacked/table_living_room_attacked.zip")
        unity_render.change_scene()
        unity_render.look_at_mesh(2., 10., 30.)
        unity_render.spawn_obj()
        main_img = unity_render.render()

        plt.figure(figsize=(7,7))
        plt.imshow(main_img)
        plt.title("unity")
        plt.show()

        plt.figure(figsize=(7,7))
        render_module.look_at_mesh(2., 10., 30.)
        plt.imshow(render_module.render(mesh_object.mesh))
        plt.title("pytorch")
        plt.show()
    finally:
        input("Press to continue")
        unity_render.despawn_obj()
        agent.delete()

