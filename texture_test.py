from saifooler.render_module import RenderModule
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt

def get_mesh_path(mesh_name):
    return f"./meshes/{mesh_name}/{mesh_name}.obj"


if __name__ == '__main__':
    # mesh_name = 'toilet'
    mesh_name = 'spaceship'

    rm = RenderModule(get_mesh_path(mesh_name))
    rm.to('cuda:0')

    """plt.figure(figsize=(7, 7))
    texture_image = rm.mesh.textures.atlas_list()[0]
    plt.imshow(texture_image.squeeze().detach().cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.show()"""


    rm.center_on_mesh(azim=180, elev=30.)
    figure = rm.show_render(return_image=False)