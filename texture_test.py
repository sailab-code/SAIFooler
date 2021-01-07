from saifooler.old.texture_module import TextureModule


def get_mesh_path(mesh_name):
    return f"./meshes/{mesh_name}/{mesh_name}.obj"


if __name__ == '__main__':
    mesh_name = 'toilet'
    mesh_1 = 'table_living_room'
    #mesh_2 = 'cube_of_power'
    mesh_2 = 'table_living_room'
    mesh_3 = 'toilet'
    meshes = [get_mesh_path(mesh_1)]
    #meshes = [get_mesh_path(mesh_1), get_mesh_path(mesh_2)]
    rm = TextureModule(meshes)
    # rm = RenderModule()
    # rm = TextureAtlasModule(get_mesh_path(mesh_name), texture_atlas_size=1024)

    rm.to('cuda:0')
    rm.show_textures()

    figure = rm.show_render(camera_params=(2, 30., 120.), return_image=False)