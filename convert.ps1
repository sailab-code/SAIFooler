#param ($fbxPath, $texturesPath, $matDefPath, $objPath)
param ($meshName, $installPath="C:\Users\enric\wkspaces\sailab_lve")

$basePath = "$installPath\Assets\Data\Objects"

$fbxPath = "$basePath\Models\${meshName}_01.fbx"
$texturesPath = "$basePath\Textures\"
$matDefPath = ".\meshes\${meshName}\${meshName}_mat_def.json"
$objPath = ".\meshes\${meshName}\${meshName}.obj"

$command = "blender --background .\blender\workspace.blend --python .\blender\scripts\convert_fbx_to_obj.py -- $fbxPath $texturesPath $matDefPath $objPath"
Write-Host EXECUTING: $command -ForegroundColor White -BackgroundColor Cyan
Write-Host `n`n 

iex $command
