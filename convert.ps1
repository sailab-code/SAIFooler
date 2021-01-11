param ($fbxPath, $texturesPath, $matDefPath, $objPath)

$command = "blender --background .\blender\workspace.blend --python .\blender\scripts\convert_fbx_to_obj.py -- $fbxPath $texturesPath $matDefPath $objPath"
Write-Host EXECUTING: $command -ForegroundColor White -BackgroundColor Cyan
Write-Host `n`n 

iex $command
