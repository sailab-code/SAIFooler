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
from saifooler.render.sailenv_module import SailenvModule

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
parser.add_argument('--eps', metavar="epsilon", type=float,
                    required=True,
                    help="Epsilon of the PGD attack")
parser.add_argument('--alpha', metavar="alpha", type=float,
                    required=True,
                    help="Alpha of the PGD attack")
parser.add_argument('--saliency', metavar='saliency', type=bool,
                    default=False, help="Wheter to use saliency for attack" )
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


def show_saliency(saliency_maps, logger, mesh_name, postfix=""):
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
    logger.experiment.add_image(f"{mesh_name}/saliency_{postfix}", blended)


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

    logger = TensorBoardLogger("./logs/pgd_imagewise_saliency")

    use_saliency = args.saliency

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

    classifier = ImageNetClassifier(used_model)
    render_module = RenderModule()

    for mesh_name, mesh_def in meshes_def.items():
        mesh_path, target_class = mesh_def["path"], mesh_def["target_class"]
        distance = mesh_def["distance"]
        orientation_elev_range = mesh_def.get("orientation_elev_range", [-90., 90.])

        mesh_descriptor = MeshDescriptor(mesh_path)

        switch_testdata = True

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

        if test_on_unity:
            # save the original mesh as a zip file
            original_zip_path = mesh_descriptor.save_to_zip()
            sailenv_noattack_evaluator = SailenvModule(agent, original_zip_path, f"{mesh_name}/sailenv", data_module,
                                                       classifier, render_module)
            sailenv_noattack_evaluator.to(device)
            sailenv_noattack_evaluator.spawn_obj()
        else:
            sailenv_noattack_evaluator = None

        saliency_estimator = SaliencyEstimator(
            mesh_descriptor.mesh,
            classifier,
            render_module,
            sailenv_noattack_evaluator,
            data_module
        )

        saliency_estimator.to(device)
        if use_saliency:
            #saliency_maps = saliency_estimator.estimate_saliency_map()
            saliency_maps = saliency_estimator.estimate_view_saliency_map()
            # show_saliency(saliency_maps[0].mean(dim=0), logger, mesh_name, "pytorch")
            if len(saliency_maps) != 1:
                # show_saliency(saliency_maps[1].mean(dim=0), logger, mesh_name, "sailenv")
                # show_saliency(saliency_maps[0].mean(dim=0) * saliency_maps[1].mean(dim=0), logger, mesh_name, "product")
                # show_saliency((saliency_maps[0].mean(dim=0) + saliency_maps[1].mean(dim=0))/2., logger, mesh_name, "mean")
                saliency_product = saliency_maps[0] * saliency_maps[1]
        else:
            saliency_maps = [None, None]

        attacker = PGDAttack(mesh_descriptor.mesh, render_module, sailenv_noattack_evaluator, classifier, epsilon, alpha,
                             mesh_name=mesh_name, saliency_maps=saliency_maps[1], saliency_threshold=0.02)
        attacker.to(device)

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            max_epochs=50,
            weights_summary=None,
            accumulate_grad_batches=data_module.number_of_batches,
            # progress_bar_refresh_rate=0,
            gpus=1,
            logger=logger
        )

        attacker.to(device)

        print(f"Attack begin against {mesh_name}")
        trainer.fit(attacker, datamodule=data_module)
        print("Testing")
        trainer.test(attacker, datamodule=data_module)
        print(f"Attack end on {mesh_name}")

        attacker.to('cpu')
        #images = attacker.render_batch(data_module.inputs)
        #post_attack_grid = Viewer3D.make_grid(images)
        #trainer.logger.experiment.add_image(f"{mesh_name}/pytorch3d_attacked", post_attack_grid.permute((2, 0, 1)))


        attacked_mesh_descriptor = mesh_descriptor.copy_to_dir(f"{logger.log_dir}/{mesh_name}_attacked", overwrite=True)

        for mat_name, new_tex in attacker.get_textures().items():
            attacked_mesh_descriptor.replace_texture(mat_name, "albedo", torch.flipud(new_tex))

        if test_on_unity:
            # prepare rendering on SAILenv

            noattack_accuracy = sailenv_noattack_evaluator.evaluate(logger).item()
            print(f"Accuracy on SAILenv before attack: {noattack_accuracy * 100}%")

            sailenv_noattack_evaluator.despawn_obj()
            # save the attacked mesh as a zip file
            attacked_zip_path = attacked_mesh_descriptor.save_to_zip()

            sailenv_attack_evaluator = SailenvModule(agent, attacked_zip_path, f"{mesh_name}/attacked_sailenv", data_module, classifier, render_module)

            sailenv_attack_evaluator.spawn_obj()
            attack_accuracy = sailenv_attack_evaluator.evaluate(logger).item()
            sailenv_attack_evaluator.despawn_obj()



            del sailenv_attack_evaluator
            del sailenv_noattack_evaluator

            print(f"Accuracy on SAILenv after attack: {attack_accuracy * 100}%")
        else:
            noattack_accuracy = 0.
            attack_accuracy = 0.

        metrics = {
            "pytorch_no_attack": attacker.accuracies['before_attack'].item(),
            "pytorch_attack": attacker.accuracies['test_accuracy'].item(),
            "sailenv_no_attack": noattack_accuracy,
            "sailenv_attack": attack_accuracy
        }

        plot = sns.barplot(
            x=list(metrics.keys()),
            y=list(metrics.values())
        )



        plot.set(ylim=(0.,1.))

        fig = plot.get_figure()

        logger.experiment.add_figure(f"{mesh_name}/summary", fig)

        with open(f"{logger.log_dir}/summary.json", "w+") as f:
            json.dump(metrics, f, indent=4)

        logger.experiment.add_text(
            "summary",
            "\n\n".join([f"**{key}**: {value:.2f}" for key, value in metrics.items()])
        )

        logger.experiment.flush()

        del attacker
        del trainer
        del data_module
        del render_module
        del classifier
        del mesh_descriptor
        del attacked_mesh_descriptor

        torch.cuda.empty_cache()

    if test_on_unity:
        agent.delete()
