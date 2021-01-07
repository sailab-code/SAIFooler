# Syntax example from the folder "blender":
# absolute/path/to/blender.exe --background workspace.blend --python scripts/convert_obj_to_fbx.py -- absolute/path/to/obj.obj absolute/path/to/fbx.fbx

# Import packages

import sys

# Import blender python
# Note: it will give an error in Pycharm, no way around it

import bpy

# Get paths from command line arguments
argv = sys.argv
# Get arguments after "--" (Blender stops parsing after that)
argv = argv[argv.index("--") + 1:]

# We need two paths
if len(argv) < 2:
    print("Error: the script needs exactly two arguments - obj_path and fbx_path")
else:
    obj_path = argv[0]
    fbx_path = argv[1]
    # Import from obj (as fast as possible)
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y', use_image_search=False)
    # Get the objects now in the scene
    imported_objs = bpy.context.selected_objects
    # Parse all imported objects
    for imported_obj in imported_objs:
        print(imported_obj)
        # Parse the all the materials assigned to the object (at least one should always be defined)
        for mat_slot in imported_obj.material_slots:
            if mat_slot.material:
                imported_obj_current_mat = mat_slot.material
                # Get nodes of the material in the node editor
                nodes = imported_obj_current_mat.node_tree.nodes
                # Get the textures nodes
                for node in nodes:
                    if node.type == 'TEX_IMAGE':
                        imported_obj_current_mat.node_tree.nodes.remove(node)
    # Export to fbx
    bpy.ops.export_scene.fbx(filepath=fbx_path, axis_forward='-Z', axis_up='Y', path_mode='COPY')


