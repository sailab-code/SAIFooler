import json
import os

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
from saifooler.utils import greyscale_heatmap, SummaryWriter
from operator import itemgetter
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
parser.add_argument('--eps', metavar="epsilon", type=float,
                    required=True,
                    help="Epsilon of the PGD attack")
parser.add_argument('--alpha', metavar="alpha", type=float,
                    required=True,
                    help="Alpha of the PGD attack")
parser.add_argument('--texture-rescale', metavar="float", type=float,
                    required=True, default=1.0,
                    help="Scale factor of the albedo textures (defaults to 1., no rescale)")
parser.add_argument('--saliency', action="store_true",
                    help="Wheter to use saliency for attack")
parser.add_argument('--saliency-threshold', metavar="alpha", type=float,
                    default=0.02, help="Threshold for constructing saliency map")
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

    agent.change_scene("object_view/scene")
    return agent


def experiment(exp_name, mesh_def, params_dict, args, log_dir="logs", switch_testdata=False):
    eps, alpha, model_name, use_saliency = itemgetter('eps', 'alpha', 'model', 'saliency')(params_dict)

    dev, use_cuda = args.device, args.cuda
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    saliency_threshold = params_dict['saliency_threshold'] if use_saliency else -1

    texture_rescale = params_dict['texture_rescale']

    if model_name == "inception":
        used_model = models.inception_v3(pretrained=True).to(device)
    elif model_name == "mobilenet":
        used_model = models.mobilenet_v2(pretrained=True).to(device)
    else:
        sys.exit("Wrong model!")

    logger = TensorBoardLogger(f"{log_dir}/pgd_attack", name=exp_name)


    os.makedirs(logger.log_dir, exist_ok=True)
    with open(f"{logger.log_dir}/params.json", "w+") as f:
        json.dump(params_dict, f, indent=4)

    logger.experiment.add_text(
        "hparams",
        "\n\n".join([f"**{key}**: {value}" for key, value in params_dict.items()])
    )

    agent = generate_agent(args)

    try:
        classifier = ImageNetClassifier(used_model)
        render_module = RenderModule()
        sailenv_module = SailenvModule(agent, render_module.lights)

        mesh_name = mesh_def["name"]
        mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
        distance = mesh_def["distance"]
        orientation_elev_range = mesh_def.get("orientation_elev_range", [-90., 90.])

        mesh_descriptor = MeshDescriptor(mesh_path).copy_to_dir(f"{logger.log_dir}/{mesh_name}_attacked",
                                                                overwrite=True)


        for mat_name, mat in mesh_descriptor.textures_path.items():
            mesh_descriptor.rescale_texture(mat_name, "albedo", texture_rescale)

        if switch_testdata:
            datamodule = MultipleViewModule(
                target_class, distance,
                orientation_elev_range=orientation_elev_range,
                orientation_elev_steps=6,
                orientation_azim_steps=5,
                light_azim_range=(0., 0.),
                light_azim_steps=1,
                light_elev_range=(50., 90.),
                light_elev_steps=1,
                batch_size=30)
        else:
            datamodule = MultipleViewModule(
                target_class, distance,
                orientation_elev_range=orientation_elev_range,
                orientation_elev_steps=10,
                orientation_azim_steps=6,
                light_azim_range=(0., 0.),
                light_azim_steps=1,
                light_elev_range=(50., 90.),
                light_elev_steps=2,
                batch_size=30)

        datamodule.setup()

        saliency_estimator = SaliencyEstimator(
            mesh_descriptor,
            classifier,
            render_module,
            sailenv_module,
            datamodule
        )
        saliency_estimator.to(device)

        if use_saliency:
            view_saliency_maps = saliency_estimator.estimate_view_saliency_map()
        else:
            view_saliency_maps = [None, None]

        attacker = PGDAttack(mesh_descriptor, render_module, sailenv_module, classifier, eps, alpha, datamodule,
                             saliency_maps=view_saliency_maps[1], saliency_threshold=saliency_threshold)

        attacker.to(device)

        monitor_metric = f'{mesh_name}_attacked/sailenv_accuracy'

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            max_epochs=1000,
            weights_summary=None,
            accumulate_grad_batches=datamodule.number_of_batches,
            check_val_every_n_epoch=5,
            # progress_bar_refresh_rate=0,
            gpus=1,
            callbacks=[EarlyStopping(monitor=monitor_metric, mode='min', patience=30),
                       ModelCheckpoint(monitor=monitor_metric, mode='min', filename=mesh_name)],
            logger=logger
        )

        # test before attack
        before_attack_results = trainer.test(attacker, datamodule=datamodule)[0]

        # print("randomly initializing weights")
        # attacker.random_initialize_delta()

        print(f"Attack begin against {mesh_name}")
        trainer.fit(attacker, datamodule=datamodule)

        print("Testing")
        after_attack_results = trainer.test(attacker, datamodule=datamodule, ckpt_path='best')[0]

        print(f"Attack end on {mesh_name}")
        attacker.to('cpu')

        metrics = {
            "pytorch_no_attack": before_attack_results[f"{mesh_name}_attacked/pytorch3d_accuracy"],
            "pytorch_attack": after_attack_results[f"{mesh_name}_attacked/pytorch3d_accuracy"],
            "sailenv_no_attack": before_attack_results[f"{mesh_name}_attacked/sailenv_accuracy"],
            "sailenv_attack": after_attack_results[f"{mesh_name}_attacked/sailenv_accuracy"]
        }

        with SummaryWriter(logger.log_dir) as w:
            w.add_hparams(params_dict, metrics)

        plot = sns.barplot(
            x=list(metrics.keys()),
            y=list(metrics.values())
        )
        plot.set(ylim=(0., 1.))
        fig = plot.get_figure()
        logger.experiment.add_figure(f"{mesh_name}_attacked/summary", fig)

        with open(f"{logger.log_dir}/summary.json", "w+") as f:
            json.dump(metrics, f, indent=4)

        logger.experiment.add_text(
            "summary",
            "\n\n".join([f"**{key}**: {value:.2f}" for key, value in metrics.items()])
        )

        agent = generate_agent(args)

        try:
            classifier = ImageNetClassifier(used_model)
            render_module = RenderModule()
            sailenv_module = SailenvModule(agent, render_module.lights)

            mesh_name = mesh_def["name"]
            mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
            distance = mesh_def["distance"]
            orientation_elev_range = mesh_def.get("orientation_elev_range", [-90., 90.])

            mesh_descriptor = MeshDescriptor(mesh_path).copy_to_dir(f"{logger.log_dir}/{mesh_name}_attacked",
                                                                    overwrite=True)


            for mat_name, mat in mesh_descriptor.textures_path.items():
                mesh_descriptor.rescale_texture(mat_name, "albedo", texture_rescale)

            if switch_testdata:
                datamodule = MultipleViewModule(
                    target_class, distance,
                    orientation_elev_range=orientation_elev_range,
                    orientation_elev_steps=6,
                    orientation_azim_steps=5,
                    light_azim_range=(0., 0.),
                    light_azim_steps=1,
                    light_elev_range=(50., 90.),
                    light_elev_steps=1,
                    batch_size=60)
            else:
                datamodule = MultipleViewModule(
                    target_class, distance,
                    orientation_elev_range=orientation_elev_range,
                    orientation_elev_steps=10,
                    orientation_azim_steps=6,
                    light_azim_range=(0., 0.),
                    light_azim_steps=1,
                    light_elev_range=(75., 90.),
                    light_elev_steps=1,
                    batch_size=60)

            datamodule.setup()

            saliency_estimator = SaliencyEstimator(
                mesh_descriptor,
                classifier,
                render_module,
                sailenv_module,
                datamodule
            )
            saliency_estimator.to(device)

            if use_saliency:
                view_saliency_maps = saliency_estimator.estimate_view_saliency_map()
            else:
                view_saliency_maps = [None, None]

            attacker = PGDAttack(mesh_descriptor, render_module, sailenv_module, classifier, eps, alpha, datamodule,
                                 saliency_maps=view_saliency_maps[1], saliency_threshold=saliency_threshold)

            attacker.to(device)

            monitor_metric = f'{mesh_name}_attacked/sailenv_accuracy'

            trainer = pl.Trainer(
                num_sanity_val_steps=0,
                max_epochs=1000,
                weights_summary=None,
                accumulate_grad_batches=datamodule.number_of_batches,
                check_val_every_n_epoch=5,
                # progress_bar_refresh_rate=0,
                gpus=1,
                callbacks=[EarlyStopping(monitor=monitor_metric, mode='min', patience=30),
                           ModelCheckpoint(monitor=monitor_metric, mode='min', filename=mesh_name)],
                logger=logger
            )

            # test before attack
            before_attack_results = trainer.test(attacker, datamodule=datamodule)[0]

            # print("randomly initializing weights")
            # attacker.random_initialize_delta()

            print(f"Attack begin against {mesh_name}")
            trainer.fit(attacker, datamodule=datamodule)

            print("Testing")
            after_attack_results = trainer.test(attacker, datamodule=datamodule, ckpt_path='best')[0]

            print(f"Attack end on {mesh_name}")
            attacker.to('cpu')

            metrics = {
                "pytorch_no_attack": before_attack_results[f"{mesh_name}_attacked/pytorch3d_accuracy"],
                "pytorch_attack": after_attack_results[f"{mesh_name}_attacked/pytorch3d_accuracy"],
                "sailenv_no_attack": before_attack_results[f"{mesh_name}_attacked/sailenv_accuracy"],
                "sailenv_attack": after_attack_results[f"{mesh_name}_attacked/sailenv_accuracy"]
            }

            with SummaryWriter(logger.log_dir) as w:
                w.add_hparams(params_dict, metrics)

            plot = sns.barplot(
                x=list(metrics.keys()),
                y=list(metrics.values())
            )
            plot.set(ylim=(0., 1.))
            fig = plot.get_figure()
            logger.experiment.add_figure(f"{mesh_name}_attacked/summary", fig)

            with open(f"{logger.log_dir}/summary.json", "w+") as f:
                json.dump(metrics, f, indent=4)

            logger.experiment.add_text(
                "summary",
                "\n\n".join([f"**{key}**: {value:.2f}" for key, value in metrics.items()])
            )

            # logger.experiment.add_hparams(params_dict, metrics)
            # log to Tensorboard HParams

            logger.experiment.flush()

            del attacker
            del trainer
            del datamodule
            del mesh_descriptor

            torch.cuda.empty_cache()
        finally:
            agent.delete()
            print(f"Experiment {exp_name} completed! \n\n\n")


if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.classifier
    epsilon = args.eps
    alpha = args.alpha
    texture_rescale = args.texture_rescale

    meshes_json_path = args.meshes_definition

    with open(meshes_json_path) as meshes_file:
        meshes_def = json.load(meshes_file)

    use_saliency = args.saliency
    saliency_threshold = args.saliency_threshold

    params_dict = {
        "eps": epsilon,
        "alpha": alpha,
        "model": model_name,
        "saliency": use_saliency,
        "texture_rescale": texture_rescale
    }

    if use_saliency:
        params_dict["saliency_threshold"] = saliency_threshold

    for mesh_name, mesh_def in meshes_def.items():
        mesh_def["mesh_name"] = mesh_name
        experiment("pgd_linf", mesh_def, params_dict, args, switch_testdata=True)
