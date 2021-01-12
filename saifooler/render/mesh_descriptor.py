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
import zipfile



class MeshDescriptor:
    def __init__(self, mesh_dir, obj_name=None, mtl_name=None, mat_def_name=None):

        self.mesh_dir = mesh_dir
        self.mesh_name = os.path.basename(mesh_dir)

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

        new_mesh_name = os.path.basename(new_dir)
        paths_to_copy = self.__get_files_paths() + self.__get_textures_paths()
        for file_path in paths_to_copy:
            filename = os.path.basename(file_path).replace(self.mesh_name, new_mesh_name)
            shutil.copy2(file_path, os.path.join(new_dir, filename))

        # obtain new paths
        new_paths = [
            path.replace(self.mesh_name, new_mesh_name) for path in
            [
                os.path.basename(self.obj_path),
                os.path.basename(self.mtl_path),
                os.path.basename(self.mat_def_path)
            ]
        ]

        # replace texture names in mat_def file
        with open(self.mat_def_path, "r") as src_mat_def_file:
            src_mat_def = json.load(src_mat_def_file)

        dst_mat_def = {
            mat_name: {
                tex_name: tex_path.replace(self.mesh_name, new_mesh_name)
                for tex_name, tex_path in mat.items()
            }
            for mat_name, mat in src_mat_def.items()
        }
        with open(os.path.join(new_dir, new_paths[2]), "w") as dst_mat_def_file:
            json.dump(dst_mat_def, dst_mat_def_file)

        # return a MeshDescriptor instance pointing to the new directory
        return self.__class__(
            new_dir,
            *new_paths
        )

    def get_texture(self, mat_name, texture_name) -> ImageType:
        return Image.open(self.textures_path[mat_name][texture_name])

    def replace_texture(self, mat_name, texture_name, texture: Union[ImageType, torch.Tensor]):
        if isinstance(texture, torch.Tensor):
            transform = torchvision.transforms.ToPILImage()
            tex_image = transform(texture.permute(2, 0, 1))
        elif isinstance(texture, ImageType):
            tex_image = texture
        else:
            raise ValueError("texture must be a tensor or a PIL image")

        tex_image.save(self.textures_path[mat_name][texture_name])

    def save_to_zip(self, zip_path=None):
        if zip_path is None:
            zip_path = os.path.join(
                self.mesh_dir,
                f"{os.path.basename(self.mesh_dir)}.zip")

        textures_paths = self.__get_textures_paths()
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        compression = zipfile.ZIP_DEFLATED

        with zipfile.ZipFile(zip_path, "w") as zip_obj:
            model_path = self.obj_path
            mat_def_path = self.mat_def_path
            print("Zipping " + model_path + " as " + os.path.basename(model_path))
            zip_obj.write(model_path, compress_type=compression, arcname=os.path.basename(model_path))
            print("Zipping " + mat_def_path + " as " + os.path.basename(mat_def_path))
            zip_obj.write(mat_def_path, compress_type=compression, arcname=os.path.basename(mat_def_path))
            if textures_paths is not None:
                for texture_path in textures_paths:
                    print("Zipping " + texture_path + " as " + os.path.basename(texture_path))
                    zip_obj.write(texture_path, compress_type=compression, arcname=os.path.basename(texture_path))

        return zip_path

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