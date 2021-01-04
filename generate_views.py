from saifooler.old.texture_module import RenderModule
import matplotlib.pyplot as plt

meshes = {
    # 'armchair': (2., ),
    # 'laptop': (1., ),
    # 'remote_controller': (1., ),
    # 'table_living_room': (2.5, ),
    # 'toilet': (1.5, ),
    'spaceship': (1., )
}

# distance, elevation, azimuth


def get_mesh_path(mesh_name):
    return f"./meshes/{mesh_name}/{mesh_name}.obj"


if __name__ == '__main__':
    for mesh_name, params in meshes.items():
        rm = RenderModule(get_mesh_path(mesh_name))
        rm.to('cuda:0')
        distance = params[0]

        for azim in range(0, 360, 60):
            rm.center_on_mesh(azim=azim, elev=20.)
            figure = rm.show_render(return_image=True)
            figure.savefig(f"./images/pytorch3d/{mesh_name}_{azim}.png")
            plt.close(figure)