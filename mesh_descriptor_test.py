from saifooler.render.mesh_descriptor import MeshDescriptor

if __name__ == '__main__':
    m = MeshDescriptor("./meshes/toilet")
    m.copy_to_dir("./meshes/toilet_attacked", overwrite=True)