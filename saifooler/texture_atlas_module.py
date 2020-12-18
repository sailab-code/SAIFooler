from typing import Any

import torch
from pytorch3d.io.mtl_io import make_mesh_texture_atlas
from pytorch3d.renderer import look_at_view_transform, TexturesAtlas
from pytorch3d.structures import Meshes
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader

from saifooler.render_module import RenderModule
import matplotlib.pyplot as plt

import pytorch3d.io as py3dio

class TextureAtlasModule(RenderModule):

    def __init__(self, mesh_path, renderer=None, *args, **kwargs):
        self.texture_atlas_size = kwargs.get("texture_atlas_size", 4)
        super().__init__(mesh_path, renderer)
        self.src_textures = {
            key: torch.nn.Parameter(tex_map)
            for key, tex_map in self.aux.texture_images.items()
        }
        for tex_name, tex_map in self.src_textures.items():
            self.register_parameter(tex_name, tex_map)
        self.atlas = self.build_atlas()

    def initialize_mesh(self, mesh_paths):
        self.mesh: Meshes = py3dio.load_objs_as_meshes(
            mesh_paths,
            create_texture_atlas=True,
            texture_atlas_size=self.texture_atlas_size,
            device=self.device)

    def build_atlas(self):
        texture_atlas = make_mesh_texture_atlas(
            self.aux.material_colors,
            self.src_textures,
            self.aux.face_material_names,
            self.aux.faces_textures_idx,
            self.aux.verts_uvs,
            self.aux.texture_atlas_size,
            self.aux.texture_wrap
        )

        return texture_atlas.to(self.device)

    def render(self, camera_params=None):
        if camera_params is not None:
            self.change_camera(*look_at_view_transform(*camera_params))

        new_mesh = self.mesh.clone()
        new_mesh.textures = TexturesAtlas(atlas=[self.build_atlas()])
        return self.renderer(new_mesh)[0, ..., :3]

    def training_step(self, batch, batch_idx):
        camera_params, target_color = batch
        image = self.render(camera_params.squeeze(0))
        self.show_render((camera_params[0][0], camera_params[0][1], camera_params[0][2]))

        loss = torch.mean(torch.pow(image-target_color, 2))
        return loss

    def show_textures(self):
        plt.figure(figsize=(7, 7))
        for tex_name, tex_map in self.src_textures.items():
            plt.imshow(tex_map.clamp(0,1).squeeze().detach().cpu().numpy())
            plt.title(tex_name)
            plt.grid("off")
            plt.axis("off")
            plt.show()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.5)

    def on_train_start(self) -> None:
        pass
        # self.show_render()
        # self.show_textures()

    def on_train_end(self) -> None:
        pass
        #self.show_render()
        #self.show_textures()



