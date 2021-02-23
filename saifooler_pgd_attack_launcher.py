import json
import os
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
from torch.utils.tensorboard import SummaryWriter
from saifooler.saliency.saliency_estimator import SaliencyEstimator
# from saifooler.utils import greyscale_heatmap
# import torchvision.transforms.functional as TF
# from PIL import Image, ImageEnhance
# import matplotlib.pyplot as plt
from itertools import product

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

    agent.change_scene("object_view/scene")
    return agent


if __name__ == '__main__':
    args = parser.parse_args()

    dev, use_cuda = args.device, args.cuda
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(f"cuda:{dev}" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # hyperparams

    EPS = [0.1, 5, 20, 30, 50]
    ALPHA = [0.001, 0.01, 0.1, 1.0]
    CLASSIFIER = ["inception", "mobilenet"]
    SALIENCY = [True, False]
    SALIENCY_THRESH = [0.01, 0.02, 0.05, 0.5]

    for eps_, alpha_, classifier_, saliency_ in product(EPS, ALPHA, CLASSIFIER, SALIENCY,
                                                        ):

        exp_name = f"eps_{eps_}__alpha_{alpha_}__model_{classifier_}_saliency_{saliency_}"

        params_dict = {"eps": eps_,
                       "alpha": alpha_,
                       "model": classifier_,
                       "saliency": saliency_}

        if saliency_:
            saliency_thresh_ = SALIENCY_THRESH

        else:
            saliency_thresh_ = [-1]

        for s_th_ in saliency_thresh_:

            if saliency_:
                exp_name += f"_saliency_thresh_{s_th_}"
                params_dict["saliency_theshold"] = s_th_

            model_name = classifier_
            if model_name == "inception":
                used_model = models.inception_v3(pretrained=True).to(device)
            elif model_name == "mobilenet":
                used_model = models.mobilenet_v2(pretrained=True).to(device)
            else:
                sys.exit("Wrong model!")

            meshes_json_path = args.meshes_definition

            with open(meshes_json_path) as meshes_file:
                meshes_def = json.load(meshes_file)

            log_folder = os.path.join("logs", "pgd", exp_name)
            logger = TensorBoardLogger(log_folder)


            use_saliency = saliency_

            agent = generate_agent(args)
            try:
                classifier = ImageNetClassifier(used_model)
                render_module = RenderModule()
                sailenv_module = SailenvModule(agent, render_module.lights)

                for mesh_name, mesh_def in meshes_def.items():
                    mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
                    distance = mesh_def["distance"]
                    orientation_elev_range = mesh_def.get("orientation_elev_range", [-90., 90.])

                    mesh_descriptor = MeshDescriptor(mesh_path).copy_to_dir(f"{logger.log_dir}/{mesh_name}_attacked",
                                                                            overwrite=True)

                    switch_testdata = False

                    if switch_testdata:
                        data_module = MultipleViewModule(
                            target_class, distance,
                            orientation_elev_range=orientation_elev_range,
                            orientation_elev_steps=6,
                            orientation_azim_steps=5,
                            light_azim_range=(0., 0.),
                            light_azim_steps=1,
                            light_elev_range=(70., 90.),
                            light_elev_steps=1,
                            batch_size=30)
                    else:
                        data_module = MultipleViewModule(
                            target_class, distance,
                            orientation_elev_range=orientation_elev_range,
                            orientation_elev_steps=6,
                            orientation_azim_steps=15,
                            light_azim_range=(0., 0.),
                            light_azim_steps=1,
                            light_elev_range=(70., 90.),
                            light_elev_steps=3,
                            batch_size=30)

                    data_module.setup()

                    saliency_estimator = SaliencyEstimator(
                        mesh_descriptor,
                        classifier,
                        render_module,
                        sailenv_module,
                        data_module
                    )
                    saliency_estimator.to(device)

                    if use_saliency:
                        view_saliency_maps = saliency_estimator.estimate_view_saliency_map()
                    else:
                        view_saliency_maps = [None, None]

                    attacker = PGDAttack(mesh_descriptor, render_module, sailenv_module, classifier, eps_, alpha_,
                                         saliency_maps=view_saliency_maps[1], saliency_threshold=s_th_)

                    attacker.to(device)

                    trainer = pl.Trainer(
                        num_sanity_val_steps=0,
                        max_epochs=100,
                        weights_summary=None,
                        accumulate_grad_batches=data_module.number_of_batches,
                        check_val_every_n_epoch=5,
                        # progress_bar_refresh_rate=0,
                        gpus=1,
                        logger=logger
                    )

                    # test before attack
                    before_attack_results = trainer.test(attacker, datamodule=data_module)[0]

                    print(f"Attack begin against {mesh_name}")
                    trainer.fit(attacker, datamodule=data_module)

                    print("Testing")
                    after_attack_results = trainer.test(attacker, datamodule=data_module)[0]

                    print(f"Attack end on {mesh_name}")
                    attacker.to('cpu')

                    metrics = {
                        "pytorch_no_attack": before_attack_results[f"{mesh_name}_attacked/pytorch3d_test_accuracy"],
                        "pytorch_attack": after_attack_results[f"{mesh_name}_attacked/pytorch3d_test_accuracy"],
                        "sailenv_no_attack": before_attack_results[f"{mesh_name}_attacked/sailenv_test_accuracy"],
                        "sailenv_attack": after_attack_results[f"{mesh_name}_attacked/sailenv_test_accuracy"]
                    }

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

                    #logger.experiment.add_hparams(params_dict, metrics)
                    # log to Tensorboard HParams
                    with SummaryWriter(logger.log_dir) as w:
                        w.add_hparams(params_dict, metrics)

                    logger.experiment.flush()

                    del attacker
                    del trainer
                    del data_module
                    del render_module
                    del classifier
                    del mesh_descriptor

                    torch.cuda.empty_cache()
            finally:
                agent.delete()
                print(f"Experiment {exp_name} completed! \n\n\n")
