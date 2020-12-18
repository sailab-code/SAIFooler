from saifooler.render_module import RenderModule
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt

from saifooler.texture_atlas_module import TextureAtlasModule
from saifooler.texture_module import TextureModule


def get_mesh_path(mesh_name):
    return f"./meshes/{mesh_name}/{mesh_name}.obj"


if __name__ == '__main__':
    # mesh_name = 'toilet'
    mesh_name = 'cube_of_power'

    # rm = TextureModule(get_mesh_path(mesh_name))
    # rm = RenderModule(get_mesh_path(mesh_name))
    rm = TextureAtlasModule(get_mesh_path(mesh_name), texture_atlas_size=1024)

    rm.to('cuda:0')
    rm.show_textures()

    figure = rm.show_render(camera_params=(3.5, 45., 180.), return_image=False)