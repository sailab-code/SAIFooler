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
from saifooler.render.unity_evaluator import SailenvEvaluator

from pytorch_lightning.loggers import TensorBoardLogger

from saifooler.viewers.viewer import Viewer3D

parser = argparse.ArgumentParser(description="Settings for PGD Attack to obj textures")
parser.add_argument('--meshes_definition', metavar='meshes_definition', type=str,
                    required=True,
                    help="Path to a json file which defines the meshes to be attacked. "
                         "The file must contain an object with the following structure."
                         '{ "<obj_name>": { "path": "<obj_dir>", "distance": "<viewing distance>", '
                         '"target_class": <imagenet_class_id> },...} See meshes_definition.example.json for an example.'
                    )
parser.add_argument('--eps', metavar="epsilon", type=float,
                    required=True,
                    help="Epsilon of the PGD attack")
parser.add_argument('--alpha', metavar="alpha", type=float,
                    required=True,
                    help="Alpha of the PGD attack")
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
    alpha = args.alpha

    meshes_json_path = args.meshes_definition

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    logger = TensorBoardLogger("./logs/pgd")

    # register agent for SAILenv
    test_on_unity = args.unitytest
    if test_on_unity:
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
        elevation, distance = mesh_def["elevation"], mesh_def["distance"]
        mesh_descriptor = MeshDescriptor(mesh_path)
        classifier = ImageNetClassifier(used_model)
        render_module = RenderModule()
        data_module = MultipleViewModule(
            target_class, distance,
            orientation_elev_steps=5,
            orientation_azim_steps=6,
            light_azim_steps=4,
            light_elev_steps=4,
            batch_size=30)
        attacker = PGDAttack(mesh_descriptor.mesh, render_module, classifier, epsilon, alpha, mesh_name=mesh_name)
        attacker.to(device)

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            max_epochs=1,
            weights_summary=None,
            # progress_bar_refresh_rate=0,
            gpus=1,
            logger=logger
        )

        print(f"Attack begin against {mesh_name}")
        trainer.fit(attacker, datamodule=data_module)
        print("Testing")
        trainer.test(attacker, datamodule=data_module)
        print(f"Attack end on {mesh_name}")

        attacker.to('cpu')
        #images = attacker.render_batch(data_module.inputs)
        #post_attack_grid = Viewer3D.make_grid(images)
        #trainer.logger.experiment.add_image(f"{mesh_name}/pytorch3d_attacked", post_attack_grid.permute((2, 0, 1)))


        attacked_mesh_descriptor = mesh_descriptor.copy_to_dir(f"./meshes/attacks/{mesh_name}_attacked", overwrite=True)

        for mat_name, new_tex in attacker.get_textures().items():
            attacked_mesh_descriptor.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

        if test_on_unity:
            # save the attacked mesh as a zip file
            attacked_zip_path = attacked_mesh_descriptor.save_to_zip()

            # save the original mesh as a zip file
            original_zip_path = mesh_descriptor.save_to_zip()

            # prepare rendering on SAILenv
            sailenv_noattack_evaluator = SailenvEvaluator(agent, original_zip_path, f"{mesh_name}/sailenv", data_module, classifier, render_module)
            noattack_accuracy = sailenv_noattack_evaluator.evaluate(logger).item()
            print(f"Accuracy on SAILenv before attack: {noattack_accuracy * 100}%")

            sailenv_attack_evaluator = SailenvEvaluator(agent, attacked_zip_path, f"{mesh_name}/attacked_sailenv", data_module, classifier, render_module)
            attack_accuracy = sailenv_attack_evaluator.evaluate(logger).item()

            del sailenv_attack_evaluator
            del sailenv_noattack_evaluator

            print(f"Accuracy on SAILenv after attack: {attack_accuracy * 100}%")
        else:
            noattack_accuracy = 0.
            attack_accuracy = 0.

        fig = sns.barplot(
            x=[
                'pytorch_no_attack',
                'pytorch_attack',
                'sailenv_no_attack',
                'sailenv_attack'
            ],
            y=[
                attacker.accuracies['before_attack'].item(),
                attacker.accuracies['test_accuracy'].item(),
                noattack_accuracy,
                attack_accuracy
            ]
        ).get_figure()

        logger.experiment.add_figure(f"{mesh_name}/summary", fig)

        logger.experiment.flush()

        del attacker
        del trainer
        del data_module
        del render_module
        del classifier
        #del images
        #del post_attack_grid
        del mesh_descriptor
        del attacked_mesh_descriptor

        torch.cuda.empty_cache()

    if test_on_unity:
        agent.delete()
