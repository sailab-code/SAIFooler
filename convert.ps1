param ($fbxPath, $objPath, $texPath)

blender --background .\blender\workspace.blend --python .\blender\scripts\convert_fbx_to_obj.py -- $fbxPath $objPath $texPath