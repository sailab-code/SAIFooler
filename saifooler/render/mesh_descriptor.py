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

        with open(self.mat_def_path, "r") as src_mat_def_file:
            src_mat_def = json.load(src_mat_def_file)

        dst_mat_def = self.__get_new_mat_def(src_mat_def, new_mesh_name)

        # obtain new obj, mtl, mat_def paths
        new_paths = {
            file: path.replace(self.mesh_name, new_mesh_name) for file, path in
            zip(['obj', 'mtl', 'mat_def'], [
                os.path.basename(self.obj_path),
                os.path.basename(self.mtl_path),
                os.path.basename(self.mat_def_path)
            ])
        }

        # copy textures with new names
        self.__copy_textures_to_new_dir(src_mat_def, dst_mat_def, new_dir)

        # replace texture names in mat_def file
        self.__replace_tex_names_in_mat_def_file(dst_mat_def, os.path.join(new_dir, new_paths['mat_def']))

        # replace texture names in mtl file
        self.__replace_tex_names_in_mtl_file(src_mat_def, dst_mat_def, os.path.join(new_dir, new_paths['mtl']))

        # replace mtl name in obj file
        self.__replace_mtl_name_in_obj_file(new_paths['mtl'], os.path.join(new_dir, new_paths['obj']))

        # return a MeshDescriptor instance pointing to the new directory
        return self.__class__(
            new_dir,
            obj_name=new_paths['obj'],
            mtl_name=new_paths['mtl'],
            mat_def_name=new_paths['mat_def']
        )

    def __get_new_mat_def(self, src_mat_def, new_mesh_name):
        return {
                mat_name: {
                    tex_name: tex_path.replace(self.mesh_name, new_mesh_name)
                    for tex_name, tex_path in mat.items()
                }
                for mat_name, mat in src_mat_def.items()
            }

    def __copy_textures_to_new_dir(self, src_mat_def, dst_mat_def, new_dir):
        for mat_name, mat in src_mat_def.items():
            for tex_name, tex_path in mat.items():
                src_tex_path = os.path.join(self.mesh_dir, tex_path)
                dst_tex_path = os.path.join(new_dir, dst_mat_def[mat_name][tex_name])
                shutil.copy2(src_tex_path, dst_tex_path)

    def __replace_tex_names_in_mtl_file(self, src_mat_def, dst_mat_def, dst_mtl_path):
        with open(self.mtl_path, "r") as src_mtl_file:
            mtl_lines = src_mtl_file.readlines()

        curr_mat_name = None
        for idx, line in enumerate(mtl_lines):
            if line.startswith("newmtl"):
                curr_mat_name = line.split(" ")[1].strip()

            if line.startswith("map_Kd"):
                mtl_lines[idx] = line.replace(
                    src_mat_def[curr_mat_name]['albedo'],
                    dst_mat_def[curr_mat_name]['albedo']
                )

            if line.startswith("map_Bump"):
                mtl_lines[idx] = line.replace(
                    src_mat_def[curr_mat_name]['normal'],
                    dst_mat_def[curr_mat_name]['normal']
                )

        with open(dst_mtl_path, "w") as dst_mtl_file:
            dst_mtl_file.writelines(mtl_lines)

    def __replace_tex_names_in_mat_def_file(self, dst_mat_def, dst_mat_def_path):
        with open(dst_mat_def_path, "w") as dst_mat_def_file:
            json.dump(dst_mat_def, dst_mat_def_file)

    def __replace_mtl_name_in_obj_file(self, new_mtl_path, dst_obj_path):
        src_obj_file = open(self.obj_path, "r")
        obj_data = src_obj_file.read()
        src_obj_file.close()

        obj_data = obj_data.replace(os.path.basename(self.mtl_path), new_mtl_path)

        with open(dst_obj_path, "w") as dst_obj_file:
            dst_obj_file.write(obj_data)

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