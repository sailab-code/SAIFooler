# Syntax example from the folder "blender":
# absolute/path/to/blender.exe --background workspace.blend --python scripts/convert_fbx_to_obj.py -- absolute/path/to/import.fbx absolute/path/to/export.obj absolute/path/to/import_mat_def.csv

# Import packages

import sys
import os
import csv

# Import blender python
# Note: it will give an error in Pycharm, no way around it

import bpy

# Get paths from command line arguments
argv = sys.argv
# Get arguments after "--" (Blender stops parsing after that)
argv = argv[argv.index("--") + 1:]

# We need three paths
if len(argv) < 3:
    print("Error: the script needs exactly three arguments - import_path, export_path and mat_def_path")
else:
    import_path = argv[0]
    export_path = argv[1]
    mat_def_path = argv[2]
    # Parse the csv file to get which texture is associated with which material
    material_texture_table = dict()
    with open(mat_def_path) as mat_def_file:
        csv_reader = csv.reader(mat_def_file, delimiter=',')
        # For each row, use the material name as key and the textures names as value
        # Note: the third row, the metallic, we don't use because .obj doesn't support it
        for row in csv_reader:
            material_texture_table[row[0]] = [row[1], row[2], row[3]]
    if len(material_texture_table) <= 0:
        print("Warning: the material texture table derived from " + mat_def_path + " is empty")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    # Import from fbx (as fast as possible)
    bpy.ops.import_scene.fbx(filepath=import_path, use_anim=False, use_custom_props=False, use_custom_props_enum_as_string=False, use_image_search=False)
    # Get the objects now in the scene
    imported_objs = bpy.context.selected_objects
    # Parse all imported objects
    for imported_obj in imported_objs:
        print(imported_obj)
        # Get the object material (or create new if not defined, we assume the name is the same of the object itself in this case)
        imported_obj_mat = imported_obj.active_material
        if imported_obj_mat is None:
            print("Object " + imported_obj.name + " no material defined, creating a new material with name " + imported_obj.name)
            imported_obj_mat = bpy.data.materials.new(name=imported_obj.name)
            # Make sure the material is assigned to the object
            if imported_obj.data.materials:
                imported_obj.data.materials[0] = imported_obj_mat
            else:
                imported_obj.data.materials.append(imported_obj_mat)
        # Parse the all the materials assigned to the object (it's only one if we built it at previous step)
        for mat_slot in imported_obj.material_slots:
            if mat_slot.material:
                imported_obj_current_mat = mat_slot.material
                # Note: this will be skipped if there is no entry in the table for the current material
                if imported_obj_current_mat.name in material_texture_table:
                    # Make sure it uses nodes
                    imported_obj_current_mat.use_nodes = True
                    # Get nodes and links of the material in the node editor
                    nodes = imported_obj_current_mat.node_tree.nodes
                    links = imported_obj_current_mat.node_tree.links
                    # Get the material shader node
                    imported_obj_current_mat_shader = nodes.get("Principled BSDF")
                    # Get the absolute path of the textures from the table (we assume the textures are in the same folder as the .csv file)
                    imported_object_current_mat_tex_path_albedo = os.path.split(mat_def_path)[0] + "/" + material_texture_table[imported_obj_current_mat.name][0]
                    imported_object_current_mat_tex_path_normal = os.path.split(mat_def_path)[0] + "/" + material_texture_table[imported_obj_current_mat.name][1]
                    # Create Image Texture node and load the albedo texture
                    imported_obj_current_mat_tex_albedo = nodes.new("ShaderNodeTexImage")
                    imported_obj_current_mat_tex_albedo.image = bpy.data.images.load(imported_object_current_mat_tex_path_albedo)
                    imported_obj_current_mat_tex_albedo.image.colorspace_settings.name = "sRGB"
                    # Create Normal Texture node and load the normal texture
                    imported_obj_current_mat_tex_normal = nodes.new("ShaderNodeTexImage")
                    imported_obj_current_mat_tex_normal.image = bpy.data.images.load(imported_object_current_mat_tex_path_normal)
                    imported_obj_current_mat_tex_normal.image.colorspace_settings.name = "Non-Color"
                    # Generate a link between the texture color output and the material output color input
                    links.new(imported_obj_current_mat_tex_albedo.outputs["Color"], imported_obj_current_mat_shader.inputs["Base Color"])
                    # Create normal map node and connect it to the color output of the normal texture and then to the normal shader input
                    imported_obj_current_normal_map = nodes.new("ShaderNodeNormalMap")
                    links.new(imported_obj_current_mat_tex_normal.outputs["Color"], imported_obj_current_normal_map.inputs["Color"])
                    links.new(imported_obj_current_normal_map.outputs["Normal"], imported_obj_current_mat_shader.inputs["Normal"])
        # Export to obj
        bpy.ops.export_scene.obj(filepath=export_path, axis_forward='-Z', axis_up='Y', path_mode='COPY')

