# Syntax example from the folder "blender":
# python scripts/package_zip.py absolute/path/to/model.xxx absolute/path/to/mat_def.csv absolute/path/to/zip.zip
# Note: the syntax example does not include optional texture paths

# Import packages

import sys
import os
import zipfile
import csv
import json

# Get paths from command line arguments
argv = sys.argv

# We need three paths (one more args because the first is the script itself)
if len(argv) < 4:
    print("Error: the script needs exactly three arguments - model_path, mat_def_path and zip_path")
else:
    model_path = argv[1]
    mat_def_path = argv[2]
    zip_path = argv[3]
    # Parse the json file to get all the texture paths (we assume they are in the same folder of the .json file)
    textures_paths = []
    with open(mat_def_path) as mat_def_file:
        mat_def = json.load(mat_def_file)

        # For each material, use the material name as key and the textures names as value
        # Note: the third row, the metallic, we don't use because .obj doesn't support it
        for mat_name, mat in mat_def.items():
            textures_paths += [
                os.path.join(os.path.dirname(mat_def_path), tex_path)
                for tex_name, tex_path in mat.items()]

    if len(textures_paths) <= 0:
        print("Warning: there are no textures defined in " + mat_def_path)
    # Make sure the zip path exists
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    # Define the compression type
    compression = zipfile.ZIP_DEFLATED
    # Prepare the zip file
    with zipfile.ZipFile(zip_path, 'w') as zip_obj:
        # Add all files to the zip
        print("Zipping " + model_path + " as " + os.path.basename(model_path))
        zip_obj.write(model_path, compress_type=compression, arcname=os.path.basename(model_path))
        print("Zipping " + mat_def_path + " as " + os.path.basename(mat_def_path))
        zip_obj.write(mat_def_path, compress_type=compression, arcname=os.path.basename(mat_def_path))
        if textures_paths is not None:
            for texture_path in textures_paths:
                print("Zipping " + texture_path + " as " + os.path.basename(texture_path))
                zip_obj.write(texture_path, compress_type=compression, arcname=os.path.basename(texture_path))
