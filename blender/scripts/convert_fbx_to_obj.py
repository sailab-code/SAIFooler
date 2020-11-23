# Syntax example from the folder "blender":
# absolute/path/to/blender.exe --background workspace.blend --python scripts/convert_fbx_to_obj.py -- absolute/path/to/candle.fbx absolute/path/to/export.obj absolute/path/to/candle_albedo_texture.png

# Import packages

import sys
import os
# Import blender python
# Note: it will give an error in Pycharm, no way around it

import bpy

# Get paths from command line arguments
argv = sys.argv
# Get arguments after "--" (Blender stops parsing after that)
argv = argv[argv.index("--") + 1:]

# We need three paths
if len(argv) < 3:
    print("Error: the script needs at least three arguments - import_path, export_path and at least one texture_path")
else:
    import_path = argv[0]
    export_path = argv[1]
    # Note: it does support multiple texture paths
    texture_paths = argv[2:]
    print(texture_paths)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    # Import from fbx (as fast as possible)
    bpy.ops.import_scene.fbx(filepath=import_path, use_anim=False, use_custom_props=False, use_custom_props_enum_as_string=False, use_image_search=False)
    # Get the objects now in the scene
    imported_objs = bpy.context.selected_objects
    imported_objs.reverse()
    print(imported_objs)
    # Parse all imported objects
    for i, imported_obj in enumerate(imported_objs):
        print(imported_obj)
        # Get the object material (or create new if not defined)
        imported_obj_mat = imported_obj.active_material
        if imported_obj_mat is None:
            print("Object " + imported_obj.name + " material undefined, creating a new material with name " + imported_obj.name)
            imported_obj_mat = bpy.data.materials.new(name=imported_obj.name)
        # Make sure it uses nodes
        imported_obj_mat.use_nodes = True
        # Get nodes and links of the material in the node editor
        nodes = imported_obj_mat.node_tree.nodes
        links = imported_obj_mat.node_tree.links
        # Get the material shader node
        imported_obj_mat_shader = nodes.get("Principled BSDF")
        # Create Image Texture node and load the albedo texture
        # Note: this will be skipped if there is no texture defined as argument for the current object
        if i < len(texture_paths):
            imported_obj_mat_tex = nodes.new("ShaderNodeTexImage")
            imported_obj_mat_tex.image = bpy.data.images.load(texture_paths[i])
            imported_obj_mat_tex.image.colorspace_settings.name = "sRGB"
            # Generate a link between the texture color output and the material output color input
            links.new(imported_obj_mat_tex.outputs["Color"], imported_obj_mat_shader.inputs["Base Color"])
        # Make sure the material is assigned to the object
        if imported_obj.data.materials:
            imported_obj.data.materials[0] = imported_obj_mat
        else:
            imported_obj.data.materials.append(imported_obj_mat)
    # Export to obj
    bpy.ops.export_scene.obj(filepath=export_path, axis_forward='-Z', axis_up='Y')

