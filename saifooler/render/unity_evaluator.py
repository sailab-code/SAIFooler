from typing import Any

from pytorch3d.renderer import camera_position_from_spherical_angles
from sailenv.agent import Agent
import pytorch_lightning as pl
import torch

from saifooler.viewers.viewer import Viewer3D


class SailenvEvaluator(pl.LightningModule):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, agent: Agent, obj_zip,
                 mesh_name: str,
                 data_module: pl.LightningDataModule,
                 classifier,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.obj_zip = obj_zip
        self.obj_id = None
        self.data_module = data_module
        self.accuracy = pl.metrics.Accuracy()
        self.classifier = classifier
        self.mesh_name = mesh_name

    def spawn_obj(self):
        remote_zip = self.agent.send_obj_zip(self.obj_zip)
        self.obj_id = self.agent.spawn_object(f"file:{remote_zip}")

    def despawn_obj(self):
        self.agent.despawn_object(self.obj_id)

    def look_at_mesh(self, distance, elevation, azimuth):
        position = camera_position_from_spherical_angles(distance, elevation, azimuth)
        rotation = (180-elevation, azimuth, 180)

        self.agent.set_position(list(position.squeeze()))
        self.agent.set_rotation(rotation)

    def render(self):
        frame = torch.tensor(self.agent.get_frame()["main"])

        # sailenv return bgr (because opencv) so we need to permute channels
        # we also need to divide by 255 because opencv returns it on range 0..255
        frame = frame[:, :, [2, 1, 0]] / 255.

        return torch.fliplr(frame).clone()

    def evaluate(self, logger=None):
        self.spawn_obj()
        images = []

        for batch_render_inputs, batch_targets in self.data_module.test_dataloader():
            for render_input, target in zip(batch_render_inputs, batch_targets):
                self.look_at_mesh(*render_input)
                image = self.render()
                images.append(image.unsqueeze(0).to(self.classifier.device))

            images = torch.cat(images)
            class_tensor = self.classifier.classify(images)
            _, classes_predicted = class_tensor.max(1, keepdim=True)
            self.accuracy(classes_predicted, batch_targets)
            self.despawn_obj()

        if logger is not None:
            images_grid = Viewer3D.make_grid(images)
            logger.experiment.add_image(self.mesh_name, images_grid.permute((2, 0, 1)))

        return self.accuracy.compute()
