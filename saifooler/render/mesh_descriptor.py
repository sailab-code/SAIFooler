from typing import Any, Union

import torch
import torchvision
from PIL import Image
from PIL.Image import Image as ImageType
from pytorch3d.structures import Meshes
import json
import os
import pytorch3d.io as py3dio
import shutil



class MeshDescriptor:
    def __init__(self, mesh_dir, obj_name=None, mtl_name=None, mat_def_name=None):

        self.mesh_dir = mesh_dir

        self.obj_path = self.__get_path(obj_name, ".obj")
        self.mtl_path = self.__get_path(mtl_name, ".mtl")
        self.mat_def_path = self.__get_path(mat_def_name, "_mat_def.json")

        with open(self.mat_def_path, "r") as mat_def_json:
            mat_def = json.load(mat_def_json)
            self.textures_path = self.__parse_mat_def(mat_def)

        self.mesh = py3dio.load_objs_as_meshes([self.obj_path])

    def copy_to_dir(self, new_dir, overwrite=False):
        try:
            os.makedirs(new_dir, exist_ok=False)
        except FileExistsError:
            if overwrite:
                print("Overwriting directory contents")
            else:
                raise RuntimeError("Directory already exists, aborting")

        paths_to_copy = self.__get_files_paths() + self.__get_textures_paths()
        for path in paths_to_copy:
            shutil.copy2(path, new_dir)

        # return a MeshDescriptor instance pointing to the new directory
        return self.__class__(
            new_dir,
            os.path.basename(self.obj_path),
            os.path.basename(self.mtl_path),
            os.path.basename(self.mat_def_path)
        )

    def get_texture(self, mat_name, texture_name) -> ImageType:
        return Image.open(self.textures_path[mat_name][texture_name])

    def replace_texture(self, mat_name, texture_name, texture: Union[ImageType, torch.Tensor]):
        if isinstance(texture, torch.Tensor):
            transform = torchvision.transforms.ToPILImage()
            tex_image = transform(texture)
        elif isinstance(texture, ImageType):
            tex_image = texture
        else:
            raise ValueError("texture must be a tensor or a PIL image")

        tex_image.save(self.textures_path[mat_name][texture_name])

    def __get_files_paths(self):
        return [
            self.obj_path,
            self.mtl_path,
            self.mat_def_path
        ]

    def __get_textures_paths(self):
        return [
            tex
            for mat_name, mat in self.textures_path.items()
            for tex_name, tex in mat.items()
        ]

    def __parse_mat_def(self, mat_def):
        return {
            mat_name: {
                tex_name: os.path.join(self.mesh_dir, tex)
                for tex_name, tex in mat.items()
            }
            for mat_name, mat in mat_def.items()
        }

    def __get_path(self, filename=None, extension=None):
        if filename is not None:
            return os.path.join(self.mesh_dir, filename)
        else:
            dir_name = os.path.basename(self.mesh_dir)
            return os.path.join(self.mesh_dir, f"{dir_name}{extension}")